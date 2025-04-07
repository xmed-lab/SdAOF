def analyze_results(file_path):
    # 读取文件并过滤最后一行
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('test result'):
                continue
            if line:
                data.append(line)

    # 解析数据
    parsed_data = []
    for line in data:
        parts = [p.strip() for p in line.split(', ')]
        entry = {
            'subject_id': parts[0].split(': ')[1],
            'dice': float(parts[1].split(': ')[1]),
            'cd_hip_left': float(parts[2].split(': ')[1]),
            'emd_hip_left': float(parts[3].split(': ')[1]),
            'cd_hip_right': float(parts[4].split(': ')[1]),
            'emd_hip_right': float(parts[5].split(': ')[1])
        }
        parsed_data.append(entry)

    # 按dice降序排序
    sorted_data = sorted(parsed_data, key=lambda x: x['dice'], reverse=True)

    # 获取用户输入
    try:
        n = int(input("请输入要剔除的样本数量N: "))
        if n <= 0 or n > len(sorted_data):
            print("无效的N值")
            return
    except ValueError:
        print("请输入有效数字")
        return

    # 获取需要剔除的样本
    removed_samples = sorted_data[-n:]
    remaining_samples = sorted_data[:-n]

    # 计算均值
    metrics = ['dice', 'cd_hip_left', 'emd_hip_left', 'cd_hip_right', 'emd_hip_right']
    means = {metric: 0.0 for metric in metrics}

    for sample in remaining_samples:
        for metric in metrics:
            means[metric] += sample[metric]

    for metric in metrics:
        means[metric] /= len(remaining_samples)

    # 输出结果
    print("\n剔除的样本Subject ID：")
    print(', '.join([s['subject_id'] for s in removed_samples]))

    print("\n剔除后各项指标均值：")
    for metric in metrics:
        print(f"{metric}: {means[metric]:.4f}")


if __name__ == "__main__":
    file_path = '/home/jchenhu/code/SdAOF/scripts/Eval_result/SdAOF_best_out_res_336/test/output.txt'
    analyze_results(file_path)