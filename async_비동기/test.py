import asyncio

# 공유 변수
shared_var = 0

async def while_one():
    global shared_var
    while True:
        if shared_var == 0:
            print("While One is working")
            shared_var = 1     # 상태 변경
        await asyncio.sleep(0.1)  # 다른 작업에게 제어권 넘기기

async def while_two():
    global shared_var
    while True:
        if shared_var == 1:
            print("While Two is working")
            shared_var = 0     # 상태 변경
        await asyncio.sleep(0.1)  # 다른 작업에게 제어권 넘기기

async def main():
    task1 = asyncio.create_task(while_one())
    task2 = asyncio.create_task(while_two())

    await asyncio.gather(task1, task2)

asyncio.run(main())
