# 步骤 1: 选择一个官方的 Python 运行时作为基础镜像
FROM python:3.10-slim

# 步骤 2: 设置工作目录
WORKDIR /app

# 步骤 3: 复制并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 步骤 4: 复制项目的所有文件到工作目录
COPY . .

# 步骤 5: 暴露端口
EXPOSE 5000

# 步骤 6: 定义容器启动时要执行的命令
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]