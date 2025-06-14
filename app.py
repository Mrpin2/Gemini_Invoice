import streamlit as st
st.set_page_config(layout="wide")

from PIL import Image
import fitz
import io
import pandas as pd
import base64
import requests
import traceback
from streamlit_lottie import st_lottie
import tempfile
import os
import locale
import re
from dateutil import parser
import json

# --- Gemini Imports ---
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Literal

locale.setlocale(locale.LC_ALL, '')

# Lottie animations for better UX
hello_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845212531.json"
completed_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845303699.json"

def load_lottie_json_safe(url):
    """Loads Lottie animation JSON safely from a URL."""
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not load Lottie animation from {url}: {e}")
        return None

hello_json = load_lottie_json_safe(hello_lottie)
completed_json = load_lottie_json_safe(completed_lottie)

# Display initial Lottie animation if no files have been uploaded yet
if "files_uploaded" not in st.session_state:
    st.session_state["files_uploaded"] = False
if not st.session_state["files_uploaded"]:
    if hello_json:
        st_lottie(hello_json, height=200, key="hello")

st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor (Gemini)</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract structured finance data using **Google Gemini**.")
st.markdown("---")

# Initialize session states for storing results and processing status
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}
if "processing_status" not in st.session_state:
    st.session_state["processing_status"] = {}
if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []
if "process_triggered" not in st.session_state:
    st.session_state["process_triggered"] = False

# This key is only for the file uploader to force reset
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

# --- Placeholders for dynamic content, including the file uploader ---
file_uploader_placeholder = st.empty()

# --- Admin/API Key Config ---
st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode", type="password")
admin_unlocked = passcode == "Rajeev"

google_api_key = None
if admin_unlocked:
    st.sidebar.success("🔓 Admin access granted.")
    google_api_key = st.secrets.get("GOOGLE_API_KEY") # Ensure GOOGLE_API_KEY is set in your Streamlit secrets
    if not google_api_key:
        st.sidebar.error("GOOGLE_API_KEY missing in Streamlit secrets.")
        st.stop()
else:
    google_api_key = st.sidebar.text_input("🔑 Enter your Google Gemini API Key", type="password")
    if not google_api_key:
        st.sidebar.warning("Please enter a valid API key to continue.")
        st.stop()

try:
    genai.configure(api_key=google_api_key)
    # Using gemini-1.5-flash-latest or gemini-2.0-flash for vision capabilities and structured output
    # Adjust model name if needed based on the latest available and preferred model
    gemini_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest", # Or "gemini-2.0-flash" if available and preferred
        generation_config={"response_mime_type": "application/json"} # Default to JSON for all generations
    )
except Exception as e:
    st.error(f"Failed to initialize Gemini client. Check your API key: {e}")
    st.stop()

# --- Pydantic Models for Structured Output ---
class LineItem(BaseModel):
    description: Optional[str] = Field(None, description="Description of the item or service.")
    hsn_sac: Optional[str] = Field(None, description="HSN (goods) or SAC (services) code. 4-8 digits, e.g., '998313'.")
    quantity: Optional[float] = Field(None, description="Quantity of the item.")
    unit_price: Optional[float] = Field(None, description="Price per unit.")
    line_total: Optional[float] = Field(None, description="Total amount for this line item before tax.")

class InvoiceData(BaseModel):
    invoice_number: Optional[str] = Field(None, description="The unique invoice number.")
    date: Optional[str] = Field(None, description="The date of the invoice in DD/MM/YYYY format.")
    seller_name: Optional[str] = Field(None, description="Name of the seller/supplier.")
    gstin: Optional[str] = Field(None, description="GSTIN of the seller (15-character alphanumeric).")
    buyer_name: Optional[str] = Field(None, description="Name of the buyer/customer.")
    buyer_gstin: Optional[str] = Field(None, description="GSTIN of the buyer (15-character alphanumeric).")
    taxable_amount: Optional[float] = Field(None, description="Subtotal before taxes, numerical value.")
    cgst: Optional[float] = Field(None, description="Central Goods and Services Tax amount, numerical value.")
    sgst: Optional[float] = Field(None, description="State Goods and Services Tax amount, numerical value.")
    igst: Optional[float] = Field(None, description="Integrated Goods and Services Tax amount, numerical value.")
    total_amount: Optional[float] = Field(None, description="Total amount of the invoice, numerical value.")
    place_of_supply: Optional[str] = Field(None, description="State/City where the supply took place (e.g., 'Delhi', 'Maharashtra').")
    expense_ledger: Optional[str] = Field(None, description="Categorization of the expense (e.g., 'Professional Fees', 'Cloud Services').")
    tds_applicability: Literal["Yes", "No", "Uncertain"] = Field("Uncertain", description="Indicates if TDS is applicable.")
    tds_section: Optional[str] = Field(None, description="TDS section if applicable (e.g., '194J', '194C').")
    tds_rate: Optional[float] = Field(None, description="TDS rate in percentage (e.g., 10.0 for 10%).")
    tds_amount: Optional[float] = Field(None, description="Calculated TDS amount.")
    amount_payable: Optional[float] = Field(None, description="Total amount minus TDS amount.")
    # Line items can be added here if detailed line item extraction is desired.
    # line_items: Optional[List[LineItem]] = Field(None, description="List of individual line items on the invoice.")

# --- Core Gemini Extraction Function ---
def extract_structured_data(image_bytes: bytes, prompt: str, pydantic_schema: BaseModel):
    """
    Extracts structured data from an image using Gemini's generate_content
    with response_schema for Pydantic model output.
    """
    try:
        image_part = {
            "mime_type": "image/png", # Assuming PNG conversion
            "data": image_bytes
        }

        # The model is instructed to fill the Pydantic schema
        response = gemini_model.generate_content(
            contents=[prompt, image_part],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": pydantic_schema.schema() # Pass the Pydantic schema
            }
        )
        
        # Gemini with response_schema directly returns the parsed object if successful
        # The response.text will contain the JSON string, and it should conform to the schema
        # We parse it here to ensure it's a Pydantic object
        raw_json_str = response.text
        # print(f"Raw JSON from Gemini: {raw_json_str}") # For debugging
        
        parsed_data = pydantic_schema.parse_raw(raw_json_str)
        return parsed_data.dict(exclude_none=True) # Return as dict, excluding nulls

    except ValidationError as e:
        st.error(f"Pydantic validation error: {e.errors()}")
        return {"error": "Pydantic validation failed", "details": e.errors()}
    except genai.types.BlockedPromptException as e:
        st.error(f"Gemini API blocked content: {e.response.prompt_feedback}")
        return {"error": "Content blocked by AI safety filters."}
    except Exception as e:
        st.error(f"Error during Gemini API call: {e}")
        return {"error": f"Gemini API error: {e}"}

# --- Functions (mostly same, adapted for Gemini's output) ---
def convert_pdf_first_page(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))

def safe_float(x):
    try:
        # Handle cases where Pydantic might give None or empty string for floats
        if x is None or str(x).strip() == '':
            return 0.0
        cleaned = str(x).replace(",", "").replace("₹", "").replace("$", "").strip()
        return float(cleaned) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0

def format_currency(x):
    try:
        if isinstance(x, str) and x.startswith('₹'):
            return x
        return f"₹{safe_float(x):,.2f}"
    except:
        return "₹0.00"

def is_valid_gstin(gstin):
    if not gstin:
        return False
    cleaned = re.sub(r'[^A-Z0-9]', '', gstin.upper())
    if len(cleaned) != 15:
        return False
    # More precise GSTIN regex (PAN part adjusted to 5 letters, 4 digits, 1 letter)
    # Corrected: Z is literal, last char is alphanumeric.
    pattern = r"^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}$"
    return bool(re.match(pattern, cleaned))

def extract_gstin_from_text(text):
    """
    Extracts a GSTIN from a given text using a regex.
    This is primarily a fallback/validation, not the primary extraction method for Gemini.
    """
    if not text:
        return ""
    # Broader regex to catch potential variations, then validate with is_valid_gstin
    matches = re.findall(r'\b(\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}[Z]{1}[A-Z0-9]{1})\b', text.upper())
    for match in matches:
        if is_valid_gstin(match):
            return match
    return ""

def determine_tds_rate(expense_ledger, tds_str="", place_of_supply=""):
    if place_of_supply and place_of_supply.lower() == "foreign":
        return 0.0
    if tds_str and isinstance(tds_str, str):
        match = re.search(r'(\d+(\.\d+)?)%', tds_str)
        if match:
            return float(match.group(1))
        section_rates = { "194j": 10.0, "194c": 1.0, "194i": 10.0, "194h": 5.0, "194q": 0.1 }
        for section, rate in section_rates.items():
            if section in tds_str.lower():
                return rate
    expense_ledger = expense_ledger.lower() if expense_ledger else ""
    if "professional" in expense_ledger or "consultancy" in expense_ledger or "service" in expense_ledger:
        return 10.0
    if "contract" in expense_ledger or "work" in expense_ledger:
        return 1.0
    if "rent" in expense_ledger:
        return 10.0
    return 0.0

def determine_tds_section(expense_ledger, place_of_supply=""):
    if place_of_supply and place_of_supply.lower() == "foreign":
        return None
    expense_ledger = expense_ledger.lower() if expense_ledger else ""
    if "professional" in expense_ledger or "consultancy" in expense_ledger or "service" in expense_ledger:
        return "194J"
    elif "contract" in expense_ledger or "work" in expense_ledger:
        return "194C"
    elif "rent" in expense_ledger:
        return "194I"
    return None

# Simplified prompt as the schema guides the structure
main_prompt = (
    "Extract the following information from the Indian invoice image. "
    "Focus on accurately identifying the fields as defined by the provided schema. "
    "If an HSN/SAC code is present, extract it; otherwise, provide `null`. "
    "For 'place_of_supply', prioritize explicit mentions, then 'Ship To:', 'Bill To:', or buyer address state/city. "
    "If it's clearly a foreign transaction/export, set 'place_of_supply' to 'Foreign'. "
    "Determine 'expense_ledger' based on the nature of goods/services. "
    "For TDS, determine applicability, section, rate, and amount. "
    "Return 'NOT AN INVOICE' for documents that are clearly not invoices."
)


# Render the file uploader using the placeholder
with file_uploader_placeholder.container():
    uploaded_files = st.file_uploader(
        "📤 Upload scanned invoice PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )

    # Store uploaded files in session state to persist across reruns and allow clearing
    if uploaded_files:
        st.session_state["uploaded_files"] = uploaded_files
        st.session_state["files_uploaded"] = True
    else:
        st.session_state["uploaded_files"] = []
        st.session_state["files_uploaded"] = False

# Conditional display of buttons after file upload
if st.session_state["files_uploaded"] or st.session_state["processed_results"]:
    col_process, col_spacer, col_clear = st.columns([1, 4, 1])

    with col_process:
        if st.button("🚀 Process Invoices", help="Click to start extracting data from uploaded invoices."):
            st.session_state["process_triggered"] = True
            st.info("Processing initiated. Please wait...")

    with col_clear:
        if st.button("🗑️ Clear All Files & Reset", help="Click to clear all uploaded files and extracted data."):
            # Increment key BEFORE clearing session_state to ensure uploader reset
            st.session_state["file_uploader_key"] += 1

            # Clear all relevant session state variables explicitly
            st.session_state["files_uploaded"] = False
            st.session_state["processed_results"] = {}
            st.session_state["processing_status"] = {}
            st.session_state["summary_rows"] = []
            st.session_state["process_triggered"] = False
            st.session_state["uploaded_files"] = [] # Explicitly empty this list

            # Clear the placeholder to remove the old file uploader instance
            file_uploader_placeholder.empty()

            # This rerun will redraw everything, including a *new* file uploader with the incremented key
            st.rerun()

# Only proceed with processing if files are uploaded AND the "Process Invoices" button was clicked
if st.session_state["uploaded_files"] and st.session_state["process_triggered"]:
    total_files = len(st.session_state["uploaded_files"])
    progress_text = st.empty()
    progress_bar = st.progress(0)

    completed_count = 0

    for idx, file in enumerate(st.session_state["uploaded_files"]):
        file_name = file.name
        progress_text.text(f"Processing file: {file_name} ({idx+1}/{total_files})")
        progress_bar.progress((idx + 1) / total_files)

        if file_name in st.session_state["processed_results"]:
            completed_count += 1
            continue

        st.markdown(f"**Current File: {file_name}**")
        st.session_state["processing_status"][file_name] = "⏳ Pending..."

        temp_file_path = None
        extracted_data = None # Will store the Pydantic-parsed dictionary
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                temp_file_path = tmp.name

            with open(temp_file_path, "rb") as f:
                pdf_data = f.read()

            first_image = convert_pdf_first_page(pdf_data)

            with st.spinner(f"🧠 Extracting data from {file_name} using Google Gemini..."):
                img_buf = io.BytesIO()
                first_image.save(img_buf, format="PNG")
                img_buf.seek(0)
                image_bytes_for_gemini = img_buf.read() # Directly use bytes for Gemini

                # Call the Gemini extraction function with the Pydantic schema
                extracted_data = extract_structured_data(
                    image_bytes=image_bytes_for_gemini,
                    prompt=main_prompt,
                    pydantic_schema=InvoiceData # Pass the Pydantic model
                )

                if extracted_data is None or extracted_data.get("error"):
                    if extracted_data and "not an invoice" in str(extracted_data.get("details", "")).lower():
                        result_row = {
                            "File Name": file_name,
                            "Invoice Number": "NOT AN INVOICE",
                            "Date": "", "Seller Name": "", "Seller GSTIN": "",
                            "HSN/SAC": "", "Buyer Name": "", "Buyer GSTIN": "",
                            "Expense Ledger": "", "Taxable Amount": 0.0,
                            "CGST": 0.0, "SGST": 0.0, "IGST": 0.0, "Total Amount": 0.0,
                            "TDS Applicability": "N/A", "TDS Section": None,
                            "TDS Rate": 0.0, "TDS Amount": 0.0, "Amount Payable": 0.0,
                            "Place of Supply": "", "TDS": "",
                            "Narration": "This document was identified as not an invoice."
                        }
                    else:
                        st.warning(f"Gemini returned an error or unparsable response for {file_name}.")
                        raise ValueError(f"Gemini returned error: {extracted_data.get('details', extracted_data.get('error', 'Unknown'))}")
                else:
                    # Directly use data from the Pydantic parsed object
                    invoice_number = extracted_data.get("invoice_number")
                    date = extracted_data.get("date")
                    seller_name = extracted_data.get("seller_name")
                    
                    # GSTINs are directly extracted by the schema.
                    # Add an extra validation/fallback, though Pydantic should enforce format.
                    seller_gstin_raw = extracted_data.get("gstin")
                    seller_gstin = seller_gstin_raw if is_valid_gstin(seller_gstin_raw) else extract_gstin_from_text(str(seller_gstin_raw))

                    buyer_gstin_raw = extracted_data.get("buyer_gstin")
                    buyer_gstin = buyer_gstin_raw if is_valid_gstin(buyer_gstin_raw) else extract_gstin_from_text(str(buyer_gstin_raw))

                    hsn_sac = extracted_data.get("hsn_sac")
                    buyer_name = extracted_data.get("buyer_name")
                    expense_ledger = extracted_data.get("expense_ledger")
                    taxable_amount = safe_float(extracted_data.get("taxable_amount"))
                    cgst = safe_float(extracted_data.get("cgst"))
                    sgst = safe_float(extracted_data.get("sgst"))
                    igst = safe_float(extracted_data.get("igst"))
                    place_of_supply = extracted_data.get("place_of_supply")
                    
                    # TDS related fields are also direct from schema
                    tds_applicability = extracted_data.get("tds_applicability", "Uncertain")
                    tds_section = extracted_data.get("tds_section")
                    tds_rate = safe_float(extracted_data.get("tds_rate"))
                    tds_amount = safe_float(extracted_data.get("tds_amount")) # Pydantic should provide this
                    
                    # Recalculate if schema didn't populate for some reason or for consistency
                    if tds_rate is None or tds_amount is None:
                        tds_rate = determine_tds_rate(expense_ledger, "", place_of_supply) # Pass empty string for tds_str as it's from schema
                        tds_section = determine_tds_section(expense_ledger, place_of_supply)
                        tds_amount = round(taxable_amount * tds_rate / 100, 2) if tds_rate > 0 else 0.0


                    total_amount = safe_float(extracted_data.get("total_amount", taxable_amount + cgst + sgst + igst))
                    # Ensure total_amount is calculated if not provided by model (less likely with response_schema)
                    if total_amount == 0.0 and (taxable_amount > 0 or cgst > 0 or sgst > 0 or igst > 0):
                        total_amount = taxable_amount + cgst + sgst + igst

                    amount_payable = safe_float(extracted_data.get("amount_payable", total_amount - tds_amount))
                    # Ensure amount_payable is calculated if not provided by model
                    if amount_payable == 0.0 and total_amount > 0 and tds_amount >= 0:
                         amount_payable = total_amount - tds_amount

                    try:
                        parsed_date = parser.parse(str(date), dayfirst=True)
                        date = parsed_date.strftime("%d/%m/%Y")
                    except:
                        date = ""

                    buyer_gstin_display = buyer_gstin or "N/A"
                    narration = (
                        f"Invoice {invoice_number or 'N/A'} dated {date or 'N/A'} "
                        f"was issued by {seller_name or 'N/A'} (GSTIN: {seller_gstin or 'N/A'}, HSN/SAC: {hsn_sac or 'N/A'}) "
                        f"to {buyer_name or 'N/A'} (GSTIN: {buyer_gstin_display}), "
                        f"with a taxable amount of ₹{taxable_amount:,.2f}. "
                        f"Taxes applied - CGST: ₹{cgst:,.2f}, SGST: ₹{sgst:,.2f}, IGST: ₹{igst:,.2f}. "
                        f"Total Amount: ₹{total_amount:,.2f}. "
                        f"Place of supply: {place_of_supply or 'N/A'}. Expense: {expense_ledger or 'N/A'}. "
                        f"TDS: {tds_applicability} (Section: {tds_section or 'N/A'}) @ {tds_rate}% (₹{tds_amount:,.2f}). "
                        f"Amount Payable: ₹{amount_payable:,.2f}."
                    )

                    result_row = {
                        "File Name": file_name,
                        "Invoice Number": invoice_number,
                        "Date": date,
                        "Seller Name": seller_name,
                        "Seller GSTIN": seller_gstin,
                        "HSN/SAC": hsn_sac,
                        "Buyer Name": buyer_name,
                        "Buyer GSTIN": buyer_gstin,
                        "Expense Ledger": expense_ledger,
                        "Taxable Amount": taxable_amount,
                        "CGST": cgst,
                        "SGST": sgst,
                        "IGST": igst,
                        "Total Amount": total_amount,
                        "TDS Applicability": tds_applicability,
                        "TDS Section": tds_section,
                        "TDS Rate": tds_rate,
                        "TDS Amount": tds_amount,
                        "Amount Payable": amount_payable,
                        "Place of Supply": place_of_supply,
                        "TDS": extracted_data.get("tds", ""), # Keep a raw TDS string from model if it was captured
                        "Narration": narration,
                    }
                    st.session_state["summary_rows"].append(result_row)
                    st.session_state["processed_results"][file_name] = result_row
                    st.session_state["processing_status"][file_name] = "✅ Extracted"

                completed_count += 1
                st.success(f"Successfully processed {file_name}")

        except Exception as e:
            st.error(f"Error processing {file_name}: {e}")
            st.session_state["processing_status"][file_name] = f"❌ Error: {e}"
            st.exception(e) # Display full traceback
            
            # Add an error row to the results for consistency
            error_details = str(e)
            if isinstance(extracted_data, dict) and extracted_data.get("error"):
                error_details += f" (Details: {extracted_data.get('details', '')})"
            
            error_result_row = {
                "File Name": file_name,
                "Invoice Number": "PROCESSING ERROR",
                "Date": "", "Seller Name": "", "Seller GSTIN": "",
                "HSN/SAC": "", "Buyer Name": "", "Buyer GSTIN": "",
                "Expense Ledger": "", "Taxable Amount": 0.0,
                "CGST": 0.0, "SGST": 0.0, "IGST": 0.0, "Total Amount": 0.0,
                "TDS Applicability": "N/A", "TDS Section": None,
                "TDS Rate": 0.0, "TDS Amount": 0.0, "Amount Payable": 0.0,
                "Place of Supply": "", "TDS": "",
                "Narration": f"Error processing file: {error_details}"
            }
            st.session_state["processed_results"][file_name] = error_result_row
            st.session_state["summary_rows"].append(error_result_row) # Add to summary for display

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    progress_text.empty()
    progress_bar.empty()
    if completed_count == total_files:
        st.success("All invoices processed!")
        if completed_json:
            st_lottie(completed_json, height=200, key="completed")
    else:
        st.warning(f"Finished processing with {total_files - completed_count} errors. Check the logs above.")

# Display results table
if st.session_state["summary_rows"]:
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>📊 Extracted Invoice Summary</h3>", unsafe_allow_html=True)

    df = pd.DataFrame(st.session_state["summary_rows"])

    # Apply formatting for currency columns
    currency_cols = ["Taxable Amount", "CGST", "SGST", "IGST", "Total Amount", "TDS Amount", "Amount Payable"]
    for col in currency_cols:
        if col in df.columns:
            df[col] = df[col].apply(format_currency)

    # Display processing status alongside the table
    df["Processing Status"] = df["File Name"].apply(lambda x: st.session_state["processing_status"].get(x, "N/A"))
    
    st.dataframe(df, use_container_width=True)

    # Download button for Excel
    st.markdown("---")
    @st.cache_data
    def convert_df_to_excel(df_to_convert):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_to_convert.to_excel(writer, index=False, sheet_name='Invoice Summary')
        processed_data = output.getvalue()
        return processed_data

    excel_data = convert_df_to_excel(df)
    st.download_button(
        label="📥 Download Summary as Excel",
        data=excel_data,
        file_name="invoice_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download the extracted invoice data as an Excel spreadsheet."
    )
    
    # Also provide CSV download, adjusted for currency formatting removal
    st.markdown("---")
    download_df = pd.DataFrame(st.session_state["summary_rows"]) # Re-create DF to ensure raw numbers for CSV

    # Handle columns for CSV download - remove formatted ones, keep original numeric
    csv_download_cols = [col for col in download_df.columns if col not in ["Processing Status", "TDS"]] # exclude 'TDS' if it's just a raw string
    
    # Manually ensure numeric conversion for CSV to avoid '₹' symbols
    for col in ["Taxable Amount", "CGST", "SGST", "IGST", "Total Amount", "TDS Amount", "Amount Payable", "TDS Rate"]:
        if col in download_df.columns:
            download_df[col] = download_df[col].apply(safe_float)

    csv_data = download_df[csv_download_cols].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", csv_data, "invoice_results.csv", "text/csv")


    # Final balloons only if all processed successfully
    if completed_count == total_files and completed_count > 0:
        st.balloons()

else:
    if not st.session_state.get("uploaded_files") and not st.session_state.get("process_triggered", False) and not st.session_state.get("processed_results"):
        st.info("Upload one or more scanned invoices to get started.")
    elif st.session_state.get("uploaded_files") and not st.session_state.get("process_triggered", False):
        st.info("Files uploaded. Click 'Process Invoices' to start extraction.")
