try:
    with open('c:/Antigravity/missile_reliability_proto_v1/v1_code/debug_log_v4.txt', 'r', encoding='cp949') as f:
        for line in f:
            if 'DEBUG' in line:
                print(line.strip())
except:
    try:
        with open('c:/Antigravity/missile_reliability_proto_v1/v1_code/debug_log_v4.txt', 'r', encoding='utf-16') as f:
            for line in f:
                if 'DEBUG' in line:
                    print(line.strip())
    except Exception as e:
        print(e)
