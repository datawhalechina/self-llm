import json
import re
import pprint
import requests

def update_contributors():
    """
    Update contributors' task numbers and sort them based on contribution.
    
    This function:
    - Reads the README.md to extract tasks
    - Counts contribution points (2 for LoRA tasks, 1 for regular tasks)
    - Adds special contributor points
    - Sorts contributors by their contribution points
    - Saves the updated data to JSON file
    - Prints contributor information
    
    Returns:
        dict: Updated contributors information
    """
    # Read files
    # Note: We are now reading from README_zh.md or the new README.md depending on where the tasks are tracked.
    # For now, let's assume we still track in the main README (which will be English) or we might need to check both.
    # Given the task is to translate, I will point this to README.md (the new English one) but we need to ensure the format matches.
    # However, the original logic parsed '@name'. Let's keep it looking at README.md but be aware it might need adjustment if the format changes.
    # Actually, since I renamed README.md to README_zh.md, and the new README.md might not have all the same details yet, 
    # maybe I should point to README_zh.md for now to preserve the contributor logic if it relies on specific Chinese markers like "微调".
    # But the goal is to translate the project. Let's stick to README.md and assume we will maintain the format.
    
    with open('./README.md', 'r', encoding='utf-8') as f:
        readme = f.read()

    with open('./contributors.json', 'r', encoding='utf-8') as f:
        contributors = json.load(f)

    # Reset task counts
    keys = contributors.keys()
    for key in keys:
        contributors[key]['task_num'] = 0

    # Extract tasks
    tasks = readme.split('\n')
    tasks = [task for task in tasks if '@' in task][:-1]

    # Count points: LoRA tasks +2, regular tasks +1
    for task in tasks:
        name = task.split('@')[1].strip() # Added strip to be safe
        if name not in keys:
            continue
        # Check for keywords in English or Chinese to be robust
        if "Lora" in task or "Fine-Tuning" in task or "微调" in task:
            contributors[name]['task_num'] += 2
        else:
            contributors[name]['task_num'] += 1

    # Add special contributor points
    special_contributors = {
        '不要葱姜蒜': 300, 
        'Logan Zou': 300
    }
    for name, points in special_contributors.items():
        if name in contributors:
            contributors[name]['task_num'] += points

    # Sort by contribution points
    contributors = dict(sorted(contributors.items(), key=lambda x: x[1]['task_num'], reverse=True))

    # Save results
    with open('./contributors.json', 'w', encoding='utf-8') as f:
        json.dump(contributors, f, indent=4, ensure_ascii=False)

    # Print results
    for key, value in contributors.items():
        print(f'- {value["info"]}')
            
    return contributors


def calculate_docker_hours():
    """
    Calculate and display Docker runtime hours from CodeWithGPU API.
    Fetches data from Datawhale account, calculates total runtime hours,
    and displays sorted information about each container.
    """
    url = "https://www.codewithgpu.com/api/v1/image/home/Datawhale?page_index=1&page_size=100&username=Datawhale"

    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9", # Translated from zh-CN,zh;q=0.9
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    }

    total_hours = 0
    docker_list = []

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data['code'] == "Success":
            item_list = data['data']['list']
            for item in item_list:
                uuid = item['uuid']
                runtime_hour = item['runtime_hour']

                if runtime_hour is not None:
                    total_hours += runtime_hour
                    docker_list.append({
                        "uuid": uuid.split("/")[-1],
                        "runtime_hour": runtime_hour
                    })
        else:
            print(f"Error: {data['message']}")

    print(f"\n{'='*60}")
    print(f"{' DOCKER RUNTIME SUMMARY ':^60}")
    print(f"{'='*60}")
    print(f"{'Total Containers:':<20} {len(docker_list):>10}")
    print(f"{'Total Runtime:':<20} {total_hours:>10.1f} hours")

    if docker_list:
        print(f"{'='*60}")
        print(f"{' DOCKER CONTAINERS (Sorted by Runtime) ':^60}")
        print(f"{'='*60}")
        print(f"{'Rank':<6} {'UUID':<35} {'Runtime (hours)':>15}")
        print(f"{'-'*6}-{'-'*35}-{'-'*15}")
        
        docker_list = sorted(docker_list, key=lambda x: x['runtime_hour'], reverse=True)
        for i, item in enumerate(docker_list, 1):
            print(f"{i:<6} {item['uuid']:<35} {item['runtime_hour']:>15.1f}")
        print(f"{'='*60}")
    else:
        print("No Docker containers found.")
    
    return docker_list, total_hours


# Usage example
if __name__ == "__main__":
    update_contributors()
    calculate_docker_hours()