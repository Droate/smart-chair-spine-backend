import requests
import time
import re

BASE_URL = "http://127.0.0.1:8000"


def run_llm_test():
    print("🚀 开始 DeepSeek API 自动化实测 (涵盖A/B/C三种画像)...")
    print("-" * 60)

    # 1. 注册并登录获取 Token
    test_username = f"test_ai_{int(time.time())}"
    test_user = {"username": test_username, "password": "password123"}

    try:
        requests.post(f"{BASE_URL}/register", json=test_user)
        token_res = requests.post(f"{BASE_URL}/token", data=test_user)
        token = token_res.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}
    except Exception as e:
        print(f"❌ 获取 Token 失败，请检查后端是否启动。错误: {e}")
        return

    # 2. 定义三种典型的测试负载 (对应论文表 6-6)
    payloads = [
        {
            "type_name": "A. 极度久坐型",
            "data": {
                "total_hours": 11.5,
                "sedentary_count": 5,
                "posture_score": 60,
                "mode_distribution": {"OFFICE": 0.9, "REST": 0.1},
                "weekly_trend": [8.0, 9.0, 9.5, 10.0, 11.0, 11.5, 11.5]
            }
        },
        {
            "type_name": "B. 表现改善型",
            "data": {
                "total_hours": 6.0,
                "sedentary_count": 1,
                "posture_score": 92,
                "mode_distribution": {"OFFICE": 0.6, "REST": 0.4},
                "weekly_trend": [9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0]
            }
        },
        {
            "type_name": "C. 异常极端值 (样本少)",
            "data": {
                "total_hours": 0.5,
                "sedentary_count": 0,
                "posture_score": 100,
                "mode_distribution": {"OFFICE": 1.0},
                "weekly_trend": [0, 0, 0, 0, 0, 0, 0.5]
            }
        }
    ]

    response_times = []
    format_compliance_count = 0
    TEST_COUNT = 50

    for i in range(TEST_COUNT):
        # 轮流使用 A, B, C 三种负载进行测试
        current_payload = payloads[i % 3]
        profile_type = current_payload["type_name"]

        print(f"⏳ [{i + 1}/{TEST_COUNT}] 测试画像: {profile_type}")
        start_time = time.time()

        try:
            res = requests.post(f"{BASE_URL}/api/v1/report/analysis", json=current_payload["data"], headers=headers)
            cost_time = time.time() - start_time
            response_times.append(cost_time)

            advice = res.json().get("advice", "")
            word_count = len(advice)
            has_markdown = bool(re.search(r'[*#>`\-]', advice))

            # 格式验证：无MD且字数在合理范围内（给大模型一点容错：70~120字算合规）
            if not has_markdown and 70 <= word_count <= 120:
                format_compliance_count += 1
                status_icon = "✅ 合规"
            else:
                status_icon = f"⚠️ 不合规(字数:{word_count}, MD:{has_markdown})"

            print(f"   {status_icon} | 耗时: {cost_time:.2f}s | 字数: {word_count}")
            print(f"   💬 回复: {advice}\n")

        except Exception as e:
            print(f"   ❌ 请求失败: {e}\n")

        time.sleep(0.8)  # 稍微加大延时，让大模型喘口气，保证回答质量

    # 统计并打印最终可填入论文的数据
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        compliance_rate = (format_compliance_count / TEST_COUNT) * 100

        print("=" * 50)
        print("📊 最终测试统计报告 (请将以下数据填入论文 6.3.3 小节)")
        print("=" * 50)
        print(f"平均响应时间: {avg_time:.2f} 秒")
        print(f"格式约束遵守率: {compliance_rate:.1f} %")
        print("=" * 50)
        print("💡 你可以直接从上面的打印记录中，挑选A、B、C各一段最好的回复，复制到论文的表6-6中！")


if __name__ == "__main__":
    run_llm_test()
