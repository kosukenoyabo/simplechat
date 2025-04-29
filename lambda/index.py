# lambda/index.py
import json
import os
import requests # boto3, re, ClientError を削除し、requests をインポート
from typing import List, Dict, Any

# FastAPIエンドポイントのURLを環境変数から取得
# デプロイ時に環境変数を設定する必要があります
INFERENCE_API_ENDPOINT = os.environ.get("INFERENCE_API_ENDPOINT")

# Bedrockクライアントの初期化とリージョン抽出関数は不要なので削除
# MODEL_ID は FastAPI 側で管理するため削除

def lambda_handler(event, context):
    """
    API Gatewayからのリクエストを処理し、FastAPIエンドポイントに推論を依頼するLambda関数。

    Args:
        event: API Gatewayからのイベントデータ。
        context: Lambda実行コンテキスト。

    Returns:
        API Gatewayに返すレスポンス。
    """
    try:
        print("Received event:", json.dumps(event))

        if not INFERENCE_API_ENDPOINT:
            raise ValueError("INFERENCE_API_ENDPOINT environment variable is not set.")

        # Cognitoで認証されたユーザー情報を取得 (オプション)
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")

        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])

        print("Sending message to inference API:", message)

        # FastAPIエンドポイントへのリクエストペイロードを作成
        api_payload = {
            "message": message,
            "conversationHistory": conversation_history
        }

        print("Calling Inference API endpoint:", INFERENCE_API_ENDPOINT)

        # FastAPIエンドポイントを呼び出し
        response = requests.post(INFERENCE_API_ENDPOINT, json=api_payload)
        response.raise_for_status() # HTTPエラーがあれば例外を発生させる

        # FastAPIからのレスポンスを解析
        api_response = response.json()
        print("Inference API response:", json.dumps(api_response))

        if not api_response.get("success"):
            error_message = api_response.get("error", "Inference API returned an error.")
            raise Exception(error_message)

        # アシスタントの応答と会話履歴を取得
        assistant_response = api_response.get("response")
        updated_history = api_response.get("conversationHistory")

        if assistant_response is None or updated_history is None:
            raise Exception("Invalid response structure from Inference API.")

        # 成功レスポンスの返却
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": updated_history
            })
        }

    except requests.exceptions.RequestException as http_error:
        print(f"HTTP Error calling Inference API: {http_error}")
        return {
            "statusCode": 502, # Bad Gateway
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": f"Failed to connect to inference service: {http_error}"
            })
        }

    except Exception as error:
        print(f"Error: {error}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }
