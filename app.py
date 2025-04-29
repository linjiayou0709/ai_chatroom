import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from agents import roles

# —— 从 ENV 读取 —— #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("請先設定環境變數 OPENAI_API_KEY")

MODEL_NAME       = os.getenv("MODEL_NAME",       "gpt-4o-mini")
TEMPERATURE      = float(os.getenv("TEMPERATURE",      0.5))
MAX_TOKENS       = int(os.getenv("MAX_TOKENS",       400))
FREQ_PENALTY     = float(os.getenv("FREQ_PENALTY",     0))
PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY", 0))

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)
# … 剩下路由逻辑不用改 …


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_message = data.get("message", "").strip()
    participants = data.get("participants", [])
    custom_map   = data.get("custom", {})

    if not user_message:
        return jsonify({"error": "請輸入訊息"}), 400
    if not participants or not isinstance(participants, list):
        return jsonify({"error": "至少選一位聊天人選"}), 400

    # 把後端預設的 roles 轉成 dict
    prompt_map = { name: prompt for name, prompt in roles }
    # 用前端自定義的 custom_map 覆蓋或補充
    # custom_map 的 key 是角色名稱，value 是完整的 system_prompt 文本
    prompt_map.update(custom_map)

    responses = []
    last_content = user_message

    # 按前端勾選的順序逐一呼叫
    for name in participants:
        system_prompt = prompt_map.get(name)
        if not system_prompt:
            # 如果 roles/custom_map 都找不到，跳過這個角色
            continue

        messages = [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": last_content},
        ]
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                frequency_penalty=FREQ_PENALTY,
                presence_penalty=PRESENCE_PENALTY,
            )
            reply = resp.choices[0].message.content.strip()
        except Exception as e:
            return jsonify({"error": f"呼叫 OpenAI API 時出錯: {e}"}), 500

        responses.append({"name": name, "content": reply})
        last_content = reply

    return jsonify({"responses": responses})


if __name__ == '__main__':
    # 調試模式下禁用靜態資源快取
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True, port=5000)