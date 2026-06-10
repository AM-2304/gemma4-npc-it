// GemmaNPCClient.cpp — Unreal Engine plugin stub for Gemma4NPC
// SPDX-License-Identifier: Apache-2.0
// Add to your Unreal project and configure via Blueprint or C++

#pragma once

#include "CoreMinimal.h"
#include "Http.h"
#include "Json.h"
#include "GameFramework/Actor.h"
#include "GemmaNPCClient.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNPCResponse, const FString&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNPCError, const FString&, Error);

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class YOURGAME_API UGemmaNPCClient : public UActorComponent
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, Category = "NPC")
    FString ServerUrl = TEXT("http://localhost:8080/v1/chat/completions");

    UPROPERTY(EditAnywhere, Category = "NPC", meta = (MultiLine = true))
    FString SystemPrompt = TEXT("You are a helpful NPC. Stay in character.");

    UPROPERTY(EditAnywhere, Category = "NPC")
    int32 MaxHistoryTurns = 10;

    UPROPERTY(EditAnywhere, Category = "NPC")
    float Temperature = 1.0f;

    UPROPERTY(EditAnywhere, Category = "NPC")
    int32 MaxTokens = 150;

    UPROPERTY(BlueprintAssignable, Category = "NPC")
    FOnNPCResponse OnNPCResponse;

    UPROPERTY(BlueprintAssignable, Category = "NPC")
    FOnNPCError OnNPCError;

    UFUNCTION(BlueprintCallable, Category = "NPC")
    void GetNPCResponse(const FString& PlayerMessage);

    UFUNCTION(BlueprintCallable, Category = "NPC")
    void ClearHistory();

private:
    TArray<TPair<FString, FString>> ConversationHistory;

    void OnResponseReceived(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bSuccess);
};

// Implementation:
void UGemmaNPCClient::GetNPCResponse(const FString& PlayerMessage)
{
    // Add user message to history
    ConversationHistory.Add(TPair<FString, FString>(TEXT("user"), PlayerMessage));

    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> HttpRequest = FHttpModule::Get().CreateRequest();
    HttpRequest->SetURL(ServerUrl);
    HttpRequest->SetVerb("POST");
    HttpRequest->SetHeader("Content-Type", "application/json");

    // Build JSON payload
    TSharedPtr<FJsonObject> JsonPayload = MakeShareable(new FJsonObject());
    JsonPayload->SetNumberField("temperature", Temperature);
    JsonPayload->SetNumberField("max_tokens", MaxTokens);

    TArray<TSharedPtr<FJsonValue>> MessagesArray;

    // Add System Prompt
    TSharedPtr<FJsonObject> SystemMsg = MakeShareable(new FJsonObject());
    SystemMsg->SetStringField("role", "system");
    SystemMsg->SetStringField("content", SystemPrompt);
    MessagesArray.Add(MakeShareable(new FJsonValueObject(SystemMsg)));

    // Add History
    int32 StartIdx = FMath::Max(0, ConversationHistory.Num() - MaxHistoryTurns);
    for (int32 i = StartIdx; i < ConversationHistory.Num(); ++i)
    {
        TSharedPtr<FJsonObject> HistMsg = MakeShareable(new FJsonObject());
        HistMsg->SetStringField("role", ConversationHistory[i].Key);
        HistMsg->SetStringField("content", ConversationHistory[i].Value);
        MessagesArray.Add(MakeShareable(new FJsonValueObject(HistMsg)));
    }

    JsonPayload->SetArrayField("messages", MessagesArray);

    FString RequestBody;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&RequestBody);
    FJsonSerializer::Serialize(JsonPayload.ToSharedRef(), Writer);

    HttpRequest->SetContentAsString(RequestBody);
    HttpRequest->OnProcessRequestComplete().BindUObject(this, &UGemmaNPCClient::OnResponseReceived);
    HttpRequest->ProcessRequest();
}

void UGemmaNPCClient::OnResponseReceived(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bSuccess)
{
    if (bSuccess && Response.IsValid() && EHttpResponseCodes::IsOk(Response->GetResponseCode()))
    {
        TSharedPtr<FJsonObject> JsonObject;
        TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Response->GetContentAsString());

        if (FJsonSerializer::Deserialize(Reader, JsonObject))
        {
            const TArray<TSharedPtr<FJsonValue>>* Choices;
            if (JsonObject->TryGetArrayField("choices", Choices) && Choices->Num() > 0)
            {
                const TSharedPtr<FJsonObject> MessageObj = (*Choices)[0]->AsObject()->GetObjectField("message");
                FString ReplyContent = MessageObj->GetStringField("content");

                // Save assistant reply
                ConversationHistory.Add(TPair<FString, FString>(TEXT("assistant"), ReplyContent));

                OnNPCResponse.Broadcast(ReplyContent);
                return;
            }
        }
    }
    
    FString ErrorMsg = Response.IsValid() ? FString::Printf(TEXT("HTTP Error: %d"), Response->GetResponseCode()) : TEXT("Request Failed");
    OnNPCError.Broadcast(ErrorMsg);
}

void UGemmaNPCClient::ClearHistory()
{
    ConversationHistory.Empty();
}
