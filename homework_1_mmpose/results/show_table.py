import json
import matplotlib.pyplot as plt

# 从JSON文件中读取数据
data_file = "E:\github\MyOpenMMLab\homework_1_mmpose\work_dir_pose\\test\\20230603_085259\\20230603_085259.json"
with open(data_file, 'r') as file:
    data = json.load(file)

# 创建一个条形图
plt.bar(range(len(data)), list(data.values()), align='center')
plt.xticks(range(len(data)), list(data.keys()), rotation='vertical')

# 添加标签
for i, v in enumerate(data.values()):
    plt.text(i, v, str(round(v, 2)), ha='center', va='bottom')

# 设置图表标题和标签
plt.title('Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Values')

# 显示图表
plt.tight_layout()
plt.savefig("E:\github\MyOpenMMLab\homework_1_mmpose\\results\pose\\result2")