import json


def calculate_sums(file_path: str) -> tuple[float, int]:
    """
    读取jsonl文件并计算所有记录中"time"字段的总和以及"new_tokens"字段的总和

    参数:
    file_path (str): jsonl文件的路径

    返回:
    tuple[float, int]: 包含两个元素的元组，第一个是time的总和，第二个是new_tokens的总和
    """
    time_sum = 0.0
    new_tokens_sum = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 解析JSON对象
                data = json.loads(line.strip())

                # 累加time字段的值
                if 'time' in data:
                    time_sum += float(data['time'])

                # 累加new_tokens字段的值
                if 'new_tokens' in data:
                    new_tokens_sum += int(data['new_tokens'])

                if 'num_new_tokens' in data:
                    new_tokens_sum += int(data['num_new_tokens'])

                if 'choices' in data:
                    new_tokens_sum += sum(data['choices'][0]['num_token'])
                    time_sum +=  sum(data['choices'][0]['wall_time'])

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        return 0.0, 0
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的jsonl格式")
        return 0.0, 0
    except Exception as e:
        print(f"错误: 处理文件 '{file_path}' 时发生意外错误: {e}")
        return 0.0, 0

    return time_sum, new_tokens_sum


# 示例用法
if __name__ == "__main__":
    file_path = r'E:\py\code_practice\TreeDecoding\benchmark\exp\llama2-temperature1\two_model_gsm8k.jsonl'  # 替换为实际的文件路径
    time_total, tokens_total = calculate_sums(file_path)
    print(f"总时间: {time_total:.2f} 秒")
    print(f"新token总数: {tokens_total}")
    print(f"tokens/sec is {tokens_total/time_total}")