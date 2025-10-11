import os
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_task.api_task_service import EmailSender, BilibiliOperator


def test_mail_sender():
    """
    Tests the basic functionality of sending an email.

    This function initializes an EmailSender, logs its SMTP server and port,
    and the sender's email and password. It includes a commented-out line
    demonstrating how to send an email with an attachment from a local path.
    """
    mail_sender = EmailSender()
    mail_sender.send_mail(
        "xiao_hong@outlook.com",
        subject="成绩告知自动化邮件",
        content="小红同学，这是你的期末成绩，相关附件已经随邮件一同发送",
        attach_local_file_path=[
            r"C:\users\12345\Desktop\第1讲 极限理论(1).pdf",
            r"C:\users\12345\Desktop\第2讲 一元积分学.pdf",
        ],
        attach_android_file_path="/sdcard/DCIM/Camera/grades.jpg",
    )


async def bilibili_viewer():
    """
    Demonstrates various operations using the BilibiliOperator.

    This asynchronous function initializes a BilibiliOperator and
    performs several actions on a specified Bilibili video (target_bvid),
    including liking/unliking, coining, and performing a "triple interaction"
    (like, coin, and favorite). It also demonstrates searching for videos
    based on keywords and printing the search results.
    """
    bilibili_operator = BilibiliOperator()

    target_bvid = "1234567"  # Bvid of the video to perform operations on

    # Like & Coin & Triple Interaction
    await bilibili_operator.like(target_bvid, True)
    await bilibili_operator.like(target_bvid, False)
    await bilibili_operator.coin(test_id=target_bvid, coin=True)
    await bilibili_operator.triple_interation(target_bvid)

    # Search for videos
    key_words = "LLM Pre-training"
    value = await bilibili_operator.search_video(key_words)
    print(f"Search results for '{key_words}' on Bilibili: {value}")


if __name__ == "__main__":
    asyncio.run(bilibili_viewer())
