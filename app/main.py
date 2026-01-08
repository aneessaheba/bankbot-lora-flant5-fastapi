# app/main.py - Complete FastAPI Application for Banking Query Classifier

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Banking77 category names
LABEL_NAMES = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay",
    "atm_support", "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance",
    "card_arrival", "card_delivery_estimate", "card_linking",
    "card_not_working", "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge",
    "exchange_rate", "exchange_via_app", "extra_charge_on_statement",
    "failed_transfer", "fiat_currency_support", "get_disposable_virtual_card",
    "get_physical_card", "getting_spare_card", "getting_virtual_card",
    "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
    "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal",
    "pending_top_up", "pending_transfer", "pin_blocked",
    "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque",
    "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal"
]

# Initialize FastAPI
app = FastAPI(
    title="Banking Query Classifier API",
    description="Classify banking customer queries into 77 categories using fine-tuned FLAN-T5 with LoRA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Request and Response models
class QueryRequest(BaseModel):
    text: str = Field(..., description="The banking query to classify", min_length=1, max_length=500)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "My card was charged twice"
            }
        }

class QueryResponse(BaseModel):
    text: str
    predicted_category: str
    confidence: Optional[str] = "high"
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "My card was charged twice",
                "predicted_category": "transaction_charged_twice",
                "confidence": "high"
            }
        }

# Global variables for model
model = None
tokenizer = None
device = None

@app.on_event("startup")
async def load_model():
    """Load the model when the API starts"""
    global model, tokenizer, device
    
    print("=" * 50)
    print("Loading Banking Query Classifier Model...")
    print("=" * 50)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Load base model
        base_model_name = "google/flan-t5-base"
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        
        # Load LoRA adapter
        model_path = "./models/lora_adapter_improved"
        print(f"Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.to(device)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("✅ Model loaded successfully!")
        print(f"Ready to classify into {len(LABEL_NAMES)} categories")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure the model files are in ./models/lora_adapter_improved/")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when API shuts down"""
    print("Shutting down API...")

def predict_category(text: str) -> dict:
    """
    Predict the category for a given banking query
    
    Args:
        text: The banking query text
        
    Returns:
        dict with 'category' and 'raw_prediction' keys
    """
    # Create prompt (same format as training!)
    prompt = f"Classify this banking query: {text}\n\nCategory:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    # Generate prediction
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            num_beams=4,
            early_stopping=True,
            do_sample=False
        )
    
    # Decode prediction
    prediction = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    
    # Match to closest label
    prediction_lower = prediction.lower().replace(" ", "_")
    best_match = None
    
    for label in LABEL_NAMES:
        if label.lower() in prediction_lower or prediction_lower in label.lower():
            best_match = label
            break
    
    final_category = best_match if best_match else prediction
    
    return {
        "category": final_category,
        "raw_prediction": prediction
    }

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Banking Query Classifier API",
        "version": "1.0.0",
        "model": "FLAN-T5 Base with LoRA",
        "categories": len(LABEL_NAMES),
        "endpoints": {
            "classify_single": "POST /predict",
            "classify_batch": "POST /predict/batch",
            "health_check": "GET /health",
            "interactive_docs": "GET /docs",
            "categories_list": "GET /categories"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None and tokenizer is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "device": str(device),
        "categories_count": len(LABEL_NAMES)
    }

@app.get("/categories")
async def list_categories():
    """Get list of all possible categories"""
    return {
        "categories": LABEL_NAMES,
        "count": len(LABEL_NAMES)
    }

@app.post("/predict", response_model=QueryResponse)
async def classify_query(request: QueryRequest):
    """
    Classify a single banking query
    
    - **text**: The banking query to classify (required)
    
    Returns the predicted category from 77 possible banking intents.
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please wait for the server to finish starting up."
        )
    
    try:
        # Get prediction
        result = predict_category(request.text)
        
        return QueryResponse(
            text=request.text,
            predicted_category=result["category"],
            confidence="high"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict/batch")
async def classify_batch(texts: List[str]):
    """
    Classify multiple banking queries at once
    
    - **texts**: List of banking queries (max 100)
    
    Returns predictions for all queries in the batch.
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please wait for the server to finish starting up."
        )
    
    # Validate batch size
    if len(texts) == 0:
        raise HTTPException(
            status_code=400, 
            detail="No texts provided"
        )
    
    if len(texts) > 100:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 100 queries per batch. Please split into multiple requests."
        )
    
    try:
        results = []
        for text in texts:
            if not text or not text.strip():
                results.append({
                    "text": text,
                    "predicted_category": "invalid_input",
                    "error": "Empty text"
                })
                continue
                
            result = predict_category(text)
            results.append({
                "text": text,
                "predicted_category": result["category"]
            })
        
        return {
            "predictions": results, 
            "count": len(results),
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Batch prediction error: {str(e)}"
        )

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )