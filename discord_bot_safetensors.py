import torch
import torch.nn.functional as F
import discord
from discord.ext import commands
import asyncio
import base64
import io
from PIL import Image
import numpy as np
import os
import argparse
from XenLiteXen3Safetensors import XenLiteXen3, DDIMScheduler
import time

# Discord Bot 설정
intents = discord.Intents.default()
intents.message_content = True

# CIFAR-10 클래스 목록
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

class ImageGenerator:
    def __init__(self, model_path, device):
        self.device = device
        
        # 모델 로드 방식 변경: safetensors 사용
        if os.path.exists(model_path) and os.path.isdir(model_path):
            try:
                # safetensors 형식으로 로드
                self.model = XenLiteXen3.from_pretrained(
                    model_path,
                    in_channels=3,
                    model_channels=48,
                    num_classes=10
                )
                print(f"Model loaded from safetensors: {model_path}")
            except Exception as e:
                print(f"Failed to load safetensors model: {e}")
                print("Creating model with random weights...")
                self.model = XenLiteXen3(in_channels=3, model_channels=48, num_classes=10)
        else:
            print(f"Warning: Model path {model_path} not found. Using random weights.")
            self.model = XenLiteXen3(in_channels=3, model_channels=48, num_classes=10)
        
        self.model.to(device)
        self.model.eval()
        
        # 스케줄러 로드 시도 (있는 경우)
        if os.path.exists(model_path) and os.path.isdir(model_path):
            scheduler_path = os.path.join(model_path, "scheduler.safetensors")
            if os.path.exists(scheduler_path):
                try:
                    self.scheduler = DDIMScheduler.from_pretrained(model_path)
                    print("Scheduler loaded from safetensors")
                except Exception as e:
                    print(f"Failed to load scheduler: {e}, using default")
                    self.scheduler = DDIMScheduler(num_timesteps=750, num_inference_steps=35)
            else:
                self.scheduler = DDIMScheduler(num_timesteps=750, num_inference_steps=35)
        else:
            self.scheduler = DDIMScheduler(num_timesteps=750, num_inference_steps=35)
        
        # 스케줄러 텐서를 올바른 디바이스로 이동
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        if hasattr(self.scheduler, 'timesteps'):
            self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    @torch.no_grad()
    def generate_image(self, class_name, num_inference_steps=35):
        # 클래스 인덱스 결정
        if class_name.lower() in [cls.lower() for cls in CIFAR10_CLASSES]:
            class_idx = [cls.lower() for cls in CIFAR10_CLASSES].index(class_name.lower())
        else:
            class_idx = np.random.randint(0, 10)
        
        batch_size = 1
        image_size = 32
        
        # 랜덤 노이즈로 시작
        image = torch.randn(batch_size, 3, image_size, image_size, device=self.device)
        
        # DDIM 스케줄러 사용
        timesteps = self.scheduler.timesteps[:num_inference_steps]
        class_labels = torch.tensor([class_idx], device=self.device)
        
        # 디노이징 루프
        for i, timestep in enumerate(timesteps):
            timestep_batch = timestep.repeat(batch_size)
            
            # 모델로 노이즈 예측
            noise_pred = self.model(image, timestep_batch, class_labels)
            
            # 이전 타임스텝으로 샘플링
            image = self.scheduler.sample_prev_timestep(
                noise_pred, timestep, image, eta=0.0
            )
        
        # 이미지 후처리: [-1, 1] -> [0, 1] -> [0, 255]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image * 255).round().byte()
        
        return image[0].cpu().numpy().transpose(1, 2, 0)

    def numpy_to_discord_file(self, image_array, filename="generated_image.png"):
        """numpy 배열을 Discord 파일로 변환"""
        image_pil = Image.fromarray(image_array)
        # 32x32를 256x256으로 업스케일 (nearest neighbor)
        image_pil = image_pil.resize((256, 256), Image.NEAREST)
        
        buffer = io.BytesIO()
        image_pil.save(buffer, format='PNG')
        buffer.seek(0)
        
        return discord.File(buffer, filename=filename)

# 전역 변수
generator = None
bot = None

class XenLiteBot(commands.Bot):
    def __init__(self):
        super().__init__(
            command_prefix='!',
            intents=intents,
            help_command=None
        )
        
    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')
        print(f'Bot is ready and serving {len(self.guilds)} guilds')
        
        # 봇 상태 설정
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="for !generate commands | XenLite Xen3"
            )
        )

    async def on_command_error(self, ctx, error):
        if isinstance(error, commands.CommandNotFound):
            await ctx.send("❌ 명령어를 찾을 수 없습니다. `!help`를 사용해보세요.")
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send("❌ 필수 인자가 누락되었습니다. `!help`를 사용해보세요.")
        else:
            await ctx.send(f"❌ 오류가 발생했습니다: {str(error)}")
            print(f"Error: {error}")

def create_bot():
    bot = XenLiteBot()
    
    @bot.command(name='generate', aliases=['gen', 'create'])
    async def generate_image(ctx, class_name: str = None, steps: int = 35):
        """
        이미지를 생성합니다.
        
        사용법:
        !generate <클래스명> [스텝수]
        !generate airplane 50
        !generate cat
        !generate (랜덤 클래스)
        """
        if steps < 15 or steps > 75:
            await ctx.send("❌ 스텝 수는 15-75 사이여야 합니다.")
            return
        
        # 클래스명 검증
        if class_name and class_name.lower() not in [cls.lower() for cls in CIFAR10_CLASSES]:
            available_classes = ", ".join(CIFAR10_CLASSES)
            embed = discord.Embed(
                title="❌ 잘못된 클래스명",
                description=f"**{class_name}**는 유효하지 않은 클래스명입니다.",
                color=0xff0000
            )
            embed.add_field(
                name="사용 가능한 클래스",
                value=available_classes,
                inline=False
            )
            await ctx.send(embed=embed)
            return
        
        # 랜덤 클래스 선택
        if not class_name:
            class_name = np.random.choice(CIFAR10_CLASSES)
        
        # 생성 시작 메시지
        loading_embed = discord.Embed(
            title="🎨 XenLite Xen3 이미지 생성 중...",
            description=f"클래스: **{class_name}**\n스텝 수: **{steps}**",
            color=0x00bfff
        )
        loading_embed.add_field(
            name="진행 상황",
            value="🔄 모델 실행 중...",
            inline=False
        )
        loading_embed.set_footer(text="잠시만 기다려주세요...")
        
        message = await ctx.send(embed=loading_embed)
        
        try:
            # 이미지 생성
            start_time = time.time()
            image_array = generator.generate_image(class_name, steps)
            generation_time = round(time.time() - start_time, 2)
            
            # Discord 파일 생성
            file = generator.numpy_to_discord_file(
                image_array, 
                f"xenlite_{class_name}_{int(time.time())}.png"
            )
            
            # 결과 임베드 생성
            result_embed = discord.Embed(
                title="✅ 이미지 생성 완료!",
                color=0x00ff00
            )
            result_embed.add_field(
                name="📊 생성 정보",
                value=f"**클래스:** {class_name}\n**스텝 수:** {steps}\n**생성 시간:** {generation_time}초",
                inline=True
            )
            result_embed.add_field(
                name="🤖 모델 정보",
                value=f"**모델:** XenLite Xen3\n**해상도:** 32x32 → 256x256\n**디바이스:** {generator.device}",
                inline=True
            )
            result_embed.set_image(url=f"attachment://{file.filename}")
            result_embed.set_footer(
                text=f"요청자: {ctx.author.display_name}", 
                icon_url=ctx.author.avatar.url if ctx.author.avatar else None
            )
            
            # 기존 메시지 편집
            await message.edit(embed=result_embed, attachments=[file])
            
        except Exception as e:
            error_embed = discord.Embed(
                title="❌ 이미지 생성 실패",
                description=f"오류: {str(e)}",
                color=0xff0000
            )
            error_embed.add_field(
                name="해결 방법",
                value="• 다른 클래스로 시도해보세요\n• 스텝 수를 조정해보세요\n• 잠시 후 다시 시도해보세요",
                inline=False
            )
            await message.edit(embed=error_embed, attachments=[])
            print(f"Generation error: {e}")
    
    @bot.command(name='classes', aliases=['cls', 'list'])
    async def list_classes(ctx):
        """사용 가능한 클래스 목록을 표시합니다."""
        class_emojis = {
            "airplane": "✈️", "automobile": "🚗", "bird": "🐦", "cat": "🐱", "deer": "🦌",
            "dog": "🐕", "frog": "🐸", "horse": "🐎", "ship": "🚢", "truck": "🚚"
        }
        
        class_list = "\n".join([
            f"{class_emojis.get(cls, '🎨')} **{cls}**" 
            for cls in CIFAR10_CLASSES
        ])
        
        embed = discord.Embed(
            title="📋 CIFAR-10 클래스 목록",
            description=class_list,
            color=0x9932cc
        )
        embed.add_field(
            name="사용법",
            value="`!generate <클래스명>`으로 이미지를 생성하세요",
            inline=False
        )
        embed.set_footer(text="총 10개의 클래스가 지원됩니다")
        
        await ctx.send(embed=embed)
    
    @bot.command(name='help', aliases=['h'])
    async def help_command(ctx):
        """도움말을 표시합니다."""
        embed = discord.Embed(
            title="🤖 XenLite Xen3 Bot 도움말",
            description="SafeTensors 기반 AI 이미지 생성 봇",
            color=0x7289da
        )
        
        embed.add_field(
            name="📝 기본 명령어",
            value="""
            `!generate <클래스> [스텝수]` - 이미지 생성
            `!classes` - 사용 가능한 클래스 목록
            `!help` - 이 도움말 표시
            `!status` - 봇 상태 확인
            """,
            inline=False
        )
        
        embed.add_field(
            name="🎨 사용 예시",
            value="""
            `!generate cat` - 고양이 이미지 생성 (기본 35스텝)
            `!generate airplane 50` - 비행기 이미지 (50스텝)
            `!generate` - 랜덤 클래스 이미지 생성
            `!gen dog 25` - 개 이미지 (25스텝, 별칭 사용)
            """,
            inline=False
        )
        
        embed.add_field(
            name="⚙️ 설정 정보",
            value=f"""
            • **스텝 범위:** 15-75 (기본값: 35)
            • **지원 클래스:** {len(CIFAR10_CLASSES)}개 (CIFAR-10)
            • **출력 해상도:** 256x256 (32x32에서 업스케일)
            • **모델:** XenLite Xen3 (SafeTensors)
            """,
            inline=False
        )
        
        embed.set_footer(text="더 빠른 생성을 원하면 스텝 수를 줄여보세요!")
        
        await ctx.send(embed=embed)
    
    @bot.command(name='status', aliases=['info'])
    async def status_command(ctx):
        """봇 상태를 표시합니다."""
        embed = discord.Embed(
            title="🔧 XenLite Xen3 봇 상태",
            color=0x00ff00
        )
        
        # 모델 정보
        embed.add_field(
            name="🤖 모델 정보", 
            value=f"**모델:** XenLite Xen3\n**채널:** 48\n**클래스:** {generator.model.num_classes}개", 
            inline=True
        )
        
        # 시스템 정보
        embed.add_field(
            name="🖥️ 시스템", 
            value=f"**디바이스:** {generator.device}\n**파이토치:** {torch.__version__}", 
            inline=True
        )
        
        # 봇 통계
        embed.add_field(
            name="📊 봇 통계", 
            value=f"**서버:** {len(bot.guilds)}개\n**사용자:** {len(bot.users)}명\n**지연시간:** {round(bot.latency * 1000)}ms", 
            inline=True
        )
        
        # 스케줄러 정보
        embed.add_field(
            name="⚡ 스케줄러",
            value=f"**타입:** DDIM\n**최대 스텝:** {generator.scheduler.num_timesteps}",
            inline=True
        )
        
        embed.set_footer(text="SafeTensors 형식으로 모델이 로드되었습니다")
        
        await ctx.send(embed=embed)
    
    @bot.command(name='batch', aliases=['multi'])
    async def batch_generate(ctx, class_name: str, count: int = 4, steps: int = 35):
        """
        여러 이미지를 한번에 생성합니다 (최대 4개)
        
        사용법: !batch <클래스> [개수] [스텝수]
        """
        if count < 1 or count > 4:
            await ctx.send("❌ 생성 개수는 1-4개 사이여야 합니다.")
            return
            
        if steps < 15 or steps > 75:
            await ctx.send("❌ 스텝 수는 15-75 사이여야 합니다.")
            return
        
        if class_name.lower() not in [cls.lower() for cls in CIFAR10_CLASSES]:
            await ctx.send(f"❌ 유효하지 않은 클래스명입니다. `!classes`를 확인해주세요.")
            return
        
        # 배치 생성 시작 메시지
        loading_embed = discord.Embed(
            title=f"🎨 배치 이미지 생성 중... ({count}개)",
            description=f"클래스: **{class_name}**\n스텝 수: **{steps}**",
            color=0x00bfff
        )
        
        message = await ctx.send(embed=loading_embed)
        
        try:
            files = []
            start_time = time.time()
            
            for i in range(count):
                # 진행 상황 업데이트
                progress_embed = discord.Embed(
                    title=f"🎨 배치 이미지 생성 중... ({i+1}/{count})",
                    description=f"클래스: **{class_name}**\n스텝 수: **{steps}**",
                    color=0x00bfff
                )
                await message.edit(embed=progress_embed)
                
                # 이미지 생성
                image_array = generator.generate_image(class_name, steps)
                file = generator.numpy_to_discord_file(
                    image_array, 
                    f"batch_{class_name}_{i+1}_{int(time.time())}.png"
                )
                files.append(file)
            
            generation_time = round(time.time() - start_time, 2)
            
            # 결과 임베드
            result_embed = discord.Embed(
                title=f"✅ 배치 생성 완료! ({count}개)",
                description=f"**클래스:** {class_name}\n**스텝 수:** {steps}\n**총 시간:** {generation_time}초\n**평균:** {round(generation_time/count, 2)}초/이미지",
                color=0x00ff00
            )
            
            await message.edit(embed=result_embed, attachments=files)
            
        except Exception as e:
            error_embed = discord.Embed(
                title="❌ 배치 생성 실패",
                description=f"오류: {str(e)}",
                color=0xff0000
            )
            await message.edit(embed=error_embed)
    
    return bot

def main():
    parser = argparse.ArgumentParser(description="XenLite Xen3 Discord Bot with SafeTensors")
    parser.add_argument("--model_path", type=str, default="./models/", 
                       help="Path to safetensors model directory")
    parser.add_argument("--token", type=str, required=True, help="Discord bot token")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cpu", "cuda"], help="Device to use")
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🚀 XenLite Xen3 Discord Bot Starting...")
    print(f"📱 Device: {device}")
    print(f"📁 Model path: {args.model_path}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    
    # 이미지 생성기 초기화
    global generator
    try:
        generator = ImageGenerator(args.model_path, device)
        print("✅ Image generator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        return
    
    # 봇 생성 및 실행
    global bot
    bot = create_bot()
    
    print("\n🤖 Available Bot Commands:")
    print("  !generate <class> [steps] - Generate single image")
    print("  !batch <class> [count] [steps] - Generate multiple images")
    print("  !classes - List available classes")
    print("  !help - Show help")
    print("  !status - Show bot status")
    print("\n🔑 Starting bot with provided token...")
    
    try:
        bot.run(args.token)
    except discord.LoginFailure:
        print("❌ Invalid Discord token. Please check your token.")
    except KeyboardInterrupt:
        print("\n⚠️ Bot stopped by user.")
    except Exception as e:
        print(f"❌ Error running bot: {e}")

if __name__ == "__main__":
    main()