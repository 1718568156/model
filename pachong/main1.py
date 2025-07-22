import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import random
garbage_class = {
    'recycle' : [
'垃圾饮料瓶',
'垃圾玻璃瓶',
'可回收易拉罐',
'可回收二手书',
    ],
    'norecycle' : [
        '剩菜剩饭',
        '不可回收烟头',
        '陶瓷垃圾',
        '垃圾一次性口罩',
        '塑料袋垃圾',
        '旧毛巾垃圾',
    ]
}
#根目录
SAVE_PATH_ROOT = "garbage_dataset"
sum_image = 0
install_number = 50

def download_images(car_class, max_per_item):
    global sum_image
    driver = webdriver.Chrome()

    for types,type_number in car_class.items():
        print(f"\n{'='*30}\n开始处理大类: {types}\n{'='*30}")
       
        
        save_path = os.path.join(SAVE_PATH_ROOT, types)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"创建文件夹: {save_path}")
        
        for keyword in type_number:
            print(f"\n--- 正在搜索具体物品: '{keyword}' ---")

            search_url = f"https://image.baidu.com/search/index?tn=baiduimage&word={keyword}&wd={keyword}"
            driver.get(search_url)

            
            image_urls = set()
            downloaded_count = 0
             # 等待页面初步加载
            time.sleep(2)

            # stall_count = 0 # “失速”计数器
            
            # while len(image_urls) < max_per_item:
            #     len_before_scroll = len(image_urls)
                
            #     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            

            img_elements = driver.find_elements(By.TAG_NAME, 'img')
            #img_elements = driver.find_elements(By.CSS_SELECTOR, ".imgitem img")
            for img in img_elements:
                src = img.get_attribute('src')
                # 筛选有效的图片 URL，排除 base64 编码的图片和空链接
                if src and src.startswith('http'):
                    image_urls.add(src)
            print(f"\r滚动加载中... 已找到 {len(image_urls)} / {max_per_item} 张不重复的图片 URL", end="")
                
                # 判断是否无法加载更多图片 (失速判断)
                # if len(image_urls) == len_before_scroll:
                #     stall_count += 1
                # else:
                #     stall_count = 0 

                # # <-- 修正：将 stall_count 判断移出 else 块，放到循环末尾 -->
                # if stall_count >= 3:
                #     print("\n页面已滚动到底部，无法加载更多图片。")
                #     break
            
            # 5. 下载图片
            print(f"\n开始下载 '{keyword}' 的图片，目标数量: {min(len(image_urls), max_per_item)}...")
            for i, url in enumerate(list(image_urls)):
                if downloaded_count >= max_per_item :
                    break
                
                try:
                    # 发送 HTTP 请求获取图片数据
                    img_response = requests.get(url, timeout=10,verify=False)
                    # 如果请求成功
                    if img_response.status_code == 200:
                        # 构造图片文件名，比如 0001.jpg, 0002.jpg
                        file_name = f"{str(sum_image + 1).zfill(4)}.jpg"
                        file_path = os.path.join(save_path, file_name)
                        
                        # 以二进制写模式保存图片
                        with open(file_path, 'wb') as f:
                            f.write(img_response.content)
                        sum_image+=1
                        downloaded_count += 1
                        print(f"  > 成功下载图片 {downloaded_count}/{max_per_item}: {file_path}")
                    else:
                        print(f"  > 下载失败，状态码: {img_response.status_code}, URL: {url}")

                except Exception as e:
                    print(f"  > 下载异常: {e}, URL: {url}")
                delay = random.uniform(0.3, 1.0)
                time.sleep(delay)
            print(f"'{keyword}' 类别下载完成，共下载 {downloaded_count} 张图片。")

    # 6. 所有任务完成后，关闭浏览器
    driver.quit()
    print("\n所有任务完成！") 


if __name__ == "__main__":
    download_images(garbage_class, install_number)