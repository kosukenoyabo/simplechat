import os
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any
import boto3
from botocore.exceptions import ClientError
import re

app = FastAPI()

# Lambda コンテキストからリージョンを抽出する関数 (lambda/index.py からコピー)
def extract_region_from_arn(arn: str) -> str:
    """
    ARNからAWSリージョンを抽出します。

    Args:
        arn: Lambda関数のARN。

    Returns:
        抽出されたリージョン名。デフォルトは 'us-east-1'。
    """
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    # 環境変数やデフォルト設定からリージョンを取得するフォールバックを追加
    return os.environ.get("AWS_REGION", "us-east-1")

# 環境変数からモデルIDを取得、なければデフォルト値を使用
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0") # デフォルト値を設定

# Bedrock クライアントを初期化（リージョンは環境変数またはデフォルトから取得）
try:
    # 環境変数 AWS_REGION が設定されていればそれを使用
    region = os.environ.get("AWS_REGION", "us-east-1")
    print(f"Initializing Bedrock client in region: {region}")
    bedrock_client = boto3.client('bedrock-runtime', region_name=region)
except Exception as e:
    print(f"Error initializing Bedrock client: {e}")
    bedrock_client = None # 初期化失敗

class ChatMessage(BaseModel):
    """
    チャットメッセージのスキーマ定義。
    """
    role: str
    content: str

class InferenceRequest(BaseModel):
    """
    推論リクエストのスキーマ定義。
    """
    message: str
    conversationHistory: List[ChatMessage] = []

class InferenceResponse(BaseModel):
    """
    推論レスポンスのスキーマ定義。
    """
    success: bool
    response: str | None = None
    conversationHistory: List[ChatMessage] | None = None
    error: str | None = None

@app.post("/infer", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest) -> InferenceResponse:
    """
    Bedrockモデルを使用して推論を実行するAPIエンドポイント。

    Args:
        request: 推論リクエストデータ。

    Returns:
        推論結果を含むレスポンス。
    """
    if bedrock_client is None:
        # Bedrockクライアントが利用できない場合、エラーを返す代わりにログを出力し、エラーレスポンスを返す
        print("Bedrock client not initialized. Cannot process inference request.")
        # FastAPIはHTTPExceptionをraiseすると処理を中断するため、ここではエラーレスポンスを返す
        return InferenceResponse(
            success=False,
            error="Bedrock client not available."
        )

    try:
        print("Processing message:", request.message)
        print("Using model:", MODEL_ID)

        # 会話履歴を構築
        messages = request.conversationHistory.copy()
        # Pydanticモデルのインスタンスとして追加
        messages.append(ChatMessage(role="user", content=request.message))

        # Bedrock 用のメッセージ形式に変換
        bedrock_messages = []
        for msg in messages:
            if msg.role == "user":
                bedrock_messages.append({
                    "role": "user",
                    "content": [{"text": msg.content}]
                })
            elif msg.role == "assistant":
                bedrock_messages.append({
                    "role": "assistant",
                    "content": [{"text": msg.content}]
                })

        # invoke_model用のリクエストペイロード
        request_payload = {
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": 512,
                "stopSequences": [],
                "temperature": 0.7,
                "topP": 0.9
            }
        }

        print("Calling Bedrock invoke_model API with payload:", json.dumps(request_payload))

        # invoke_model APIを呼び出し
        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(request_payload),
            contentType="application/json"
        )

        # レスポンスを解析
        response_body = json.loads(response['body'].read())
        print("Bedrock response:", json.dumps(response_body, default=str))

        # 応答の検証
        # response_body['output']が存在しない場合や、想定された構造でない場合のチェックを強化
        if not response_body.get('output') or not isinstance(response_body['output'], dict) or \
           not response_body['output'].get('message') or not isinstance(response_body['output']['message'], dict) or \
           not response_body['output']['message'].get('content') or not isinstance(response_body['output']['message']['content'], list) or \
           len(response_body['output']['message']['content']) == 0 or \
           not response_body['output']['message']['content'][0].get('text'):
             print("Unexpected response structure from Bedrock:", response_body) # 詳細ログ
             raise Exception("No valid response content from the model")


        # アシスタントの応答を取得
        assistant_response = response_body['output']['message']['content'][0]['text']

        # 更新された会話履歴 (Pydanticモデルのリストとして)
        updated_history = messages + [ChatMessage(role="assistant", content=assistant_response)]

        return InferenceResponse(
            success=True,
            response=assistant_response,
            # Pydanticモデルのリストを辞書のリストに変換して返す
            conversationHistory=[msg.dict() for msg in updated_history]
        )

    except ClientError as error:
        print(f"Bedrock Client Error: {error}")
        # クライアントエラーの場合もエラーレスポンスを返す
        return InferenceResponse(success=False, error=f"Bedrock API error: {error}")
    except Exception as error:
        print(f"Error: {error}")
        # その他のエラーの場合もエラーレスポンスを返す
        return InferenceResponse(success=False, error=str(error))

@app.get("/health")
async def health_check():
    """
    ヘルスチェック用エンドポイント。
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # 環境変数 PORT があればそれを使用、なければ 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 