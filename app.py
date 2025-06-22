import streamlit as st
import json
import datetime
import random
import math
import requests # For making the Gemini API call

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Restaurant Bill/KOT Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def format_number(num, use_comma):
    """Formats a number to 2 decimal places with optional comma as decimal separator."""
    if not isinstance(num, (int, float)) or math.isnan(num):
        return "0.00"
    
    # Python's locale formatting can be complex, so we'll do a simple string manipulation
    formatted_str = f"{num:.2f}"
    if use_comma:
        return formatted_str.replace('.', ',')
    return formatted_str

def get_random_time_for_meal(meal_type):
    """Generates a random datetime string for a given meal type."""
    now = datetime.datetime.now()
    
    start_hour, end_hour, start_minute, end_minute = 0, 0, 0, 0

    if meal_type == 'breakfast':
        start_hour, start_minute = 9, 0
        end_hour, end_minute = 11, 40
    elif meal_type == 'lunch':
        start_hour, start_minute = 13, 40
        end_hour, end_minute = 16, 0
    elif meal_type == 'dinner':
        start_hour, start_minute = 20, 0
        end_hour, end_minute = 22, 0
    else:
        return now.isoformat(timespec='minutes') # Fallback to current time

    total_minutes_start = start_hour * 60 + start_minute
    total_minutes_end = end_hour * 60 + end_minute
    random_total_minutes = random.randint(total_minutes_start, total_minutes_end)

    random_hour = random_total_minutes // 60
    random_minute = random_total_minutes % 60

    new_time = now.replace(hour=random_hour, minute=random_minute, second=0, microsecond=0)
    return new_time.isoformat(timespec='minutes')

def get_gemini_api_key():
    """Retrieves Gemini API key from Streamlit secrets or returns empty string."""
    try:
        # For deployment on Streamlit Cloud, use st.secrets
        # Go to 'Manage app' -> '...' menu -> 'Edit Secrets' to add GEMINI_API_KEY
        return st.secrets["GEMINI_API_KEY"]
    except:
        # Fallback for local testing or if secret is not set
        return "" # Will be filled by Canvas runtime when running in Canvas

def call_gemini_api(prompt, api_key):
    """Makes a call to the Gemini API with structured response request."""
    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    
    payload = {
        "contents": chat_history,
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "address": {"type": "STRING"},
                        "city": {"type": "STRING"},
                        "state": {"type": "STRING"},
                        "telephone": {"type": "STRING"},
                        "gstin": {"type": "STRING"}
                    },
                    "required": ["name", "address", "city", "state", "telephone", "gstin"]
                }
            }
        }
    }

    headers = {'Content-Type': 'application/json'}
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    response.raise_for_status() # Raise an exception for HTTP errors
    
    result = response.json()
    if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
        return json.loads(result["candidates"][0]["content"]["parts"][0]["text"])
    return None

# --- Initial State Setup (using st.session_state) ---
if 'initialized' not in st.session_state:
    st.session_state.restaurant = {
        'name': 'Karnataka Food Centre',
        'address': 'R. K. Puram, Sector 12',
        'city': 'New Delhi',
        'state': 'Delhi',
        'telephone': '+91 11 2617 1122',
        'gstin': '07ABCE12345FG'
    }
    st.session_state.order_info = {
        'orderNo': 'ORD-2025-001',
        'tableNo': 'A5',
        'serverName': 'Ravi',
        'orderDateTime': datetime.datetime.now().isoformat(timespec='minutes')
    }
    st.session_state.items = [
        {'id': 1, 'name': 'Masala Dosa', 'quantity': 2, 'price': 142.86, 'notes': ''},
        {'id': 2, 'name': 'Filter Coffee', 'quantity': 1, 'price': 50.00, 'notes': 'Less Sugar'}
    ]
    st.session_state.include_service_charge = True
    st.session_state.service_charge_percentage = 10
    st.session_state.tip_amount = 0
    st.session_state.discount_percentage = 0
    st.session_state.round_off_total = False
    st.session_state.gst_rate = 5
    st.session_state.bill_type = 'customer-bill'
    st.session_state.font_style = 'sans'
    st.session_state.text_style = 'normal'
    st.session_state.currency_symbol = '₹'
    st.session_state.use_comma_decimal_separator = False
    st.session_state.credit_card_last_four = ''
    st.session_state.payment_details = {
        'transactionType': 'SALE',
        'authorization': 'APPROVED',
        'paymentCode': '66402563',
        'paymentId': '9707487',
        'cardReader': 'SWIPED/CHIP'
    }
    st.session_state.display_customer_copy = False
    st.session_state.thank_you_message = 'Thank you for visiting!'
    st.session_state.promotional_message = ''
    st.session_state.include_logo = False
    st.session_state.ai_password = ''
    st.session_state.ai_unlocked = False
    st.session_state.ai_location = ''
    st.session_state.ai_meal_type = ''
    st.session_state.ai_loading = False
    st.session_state.ai_error = ''
    st.session_state.initialized = True

# --- Callbacks for Item Management ---
def add_item_callback():
    new_id = max([item['id'] for item in st.session_state.items]) + 1 if st.session_state.items else 1
    st.session_state.items.append({'id': new_id, 'name': '', 'quantity': 1, 'price': 0.0, 'notes': ''})

def remove_item_callback(item_id):
    st.session_state.items = [item for item in st.session_state.items if item['id'] != item_id]

def duplicate_item_callback(item_id):
    item_to_duplicate = next((item for item in st.session_state.items if item['id'] == item_id), None)
    if item_to_duplicate:
        new_id = max([item['id'] for item in st.session_state.items]) + 1 if st.session_state.items else 1
        st.session_state.items.append({**item_to_duplicate, 'id': new_id})

def move_item_callback(item_id, direction):
    items_list = st.session_state.items
    idx = next((i for i, item in enumerate(items_list) if item['id'] == item_id), -1)
    
    if idx == -1: return

    if direction == 'up' and idx > 0:
        items_list[idx], items_list[idx-1] = items_list[idx-1], items_list[idx]
    elif direction == 'down' and idx < len(items_list) - 1:
        items_list[idx], items_list[idx+1] = items_list[idx+1], items_list[idx]
    st.session_state.items = items_list # Update state to trigger rerun

def reset_bill_callback():
    # Re-initialize all session state variables to their defaults
    del st.session_state['initialized']
    st.rerun()

def copy_bill_text_callback():
    # Streamlit doesn't have a direct "copy to clipboard" for arbitrary text
    # We'll generate the text and put it in a temporary text_area for user to copy.
    # The actual copy logic will be handled by the "Download as Text File" or
    # instructing the user to copy from the generated markdown/text_area.
    st.session_state.show_copy_text_area = True

def download_bill_text_callback():
    # Generate the bill text content
    bill_content_markdown = generate_bill_html(
        st.session_state.restaurant,
        st.session_state.order_info,
        st.session_state.items,
        st.session_state.include_service_charge,
        st.session_state.service_charge_percentage,
        st.session_state.tip_amount,
        st.session_state.discount_percentage,
        st.session_state.round_off_total,
        st.session_state.gst_rate,
        st.session_state.bill_type,
        st.session_state.font_style,
        st.session_state.text_style,
        st.session_state.currency_symbol,
        st.session_state.use_comma_decimal_separator,
        st.session_state.credit_card_last_four,
        st.session_state.payment_details,
        st.session_state.display_customer_copy,
        st.session_state.thank_you_message,
        st.session_state.promotional_message,
        st.session_state.include_logo,
        for_download=True # Indicate it's for download to get plain text
    )
    # Provide a download button
    st.download_button(
        label="Click to Download Bill Text",
        data=bill_content_markdown,
        file_name=f"restaurant_bill_{st.session_state.order_info['orderNo']}.txt",
        mime="text/plain"
    )

def ai_password_submit_callback():
    if st.session_state.ai_password_input == 'gemini':
        st.session_state.ai_unlocked = True
        st.session_state.ai_error = ''
    else:
        st.session_state.ai_error = 'Incorrect password. Try "gemini".'
        st.session_state.ai_unlocked = False

def fetch_ai_details_callback():
    st.session_state.ai_loading = True
    st.session_state.ai_error = ''
    
    api_key = get_gemini_api_key()
    if not api_key:
        st.session_state.ai_error = "Gemini API Key not found. Please set it in Streamlit Secrets or provide it."
        st.session_state.ai_loading = False
        return

    try:
        prompt = (
            f"Give me details for 1 well-known South Indian restaurant in "
            f"{st.session_state.ai_location or 'New Delhi'} suitable for a {st.session_state.ai_meal_type or 'general'} visit. "
            f"For this restaurant, provide its name, a concise address (street, area, city, state), "
            f"a typical Indian telephone number in +91 format, and a realistic Indian GSTIN (e.g., 07ABCDE1234F1Z5). "
            f"Format the output as a JSON array containing one object with keys: 'name', 'address', 'city', 'state', 'telephone', 'gstin'."
        )
        
        restaurants_data = call_gemini_api(prompt, api_key)
        
        if restaurants_data and len(restaurants_data) > 0:
            first_restaurant = restaurants_data[0]
            st.session_state.restaurant = first_restaurant
            st.session_state.order_info['orderDateTime'] = get_random_time_for_meal(st.session_state.ai_meal_type)
        else:
            st.session_state.ai_error = 'AI could not find suitable restaurant details.'
            
    except requests.exceptions.RequestException as e:
        st.session_state.ai_error = f"Network or API error: {e}"
    except json.JSONDecodeError:
        st.session_state.ai_error = "AI returned unreadable JSON. Try again or refine prompt."
    except Exception as e:
        st.session_state.ai_error = f"An unexpected error occurred: {e}"
    finally:
        st.session_state.ai_loading = False
    
# --- Main Application Layout ---
st.title("Restaurant Bill/KOT Generator")

# --- Control Panel ---
st.sidebar.header("Controls & Customization")

with st.sidebar:
    st.subheader("Bill Type & Styling")
    st.session_state.bill_type = st.radio(
        "Select Bill Type:",
        ('customer-bill', 'kot', 'simple-receipt', 'thermal-1', 'thermal-2'),
        format_func=lambda x: {
            'customer-bill': 'Customer Bill (Detailed)',
            'kot': 'KOT (Kitchen Order Ticket)',
            'simple-receipt': 'Simple Receipt',
            'thermal-1': 'Thermal Receipt Style 1',
            'thermal-2': 'Thermal Receipt Style 2'
        }[x],
        key='bill_type_radio'
    )

    st.session_state.font_style = st.radio(
        "Select Font Style:",
        ('sans', 'serif', 'mono'),
        format_func=lambda x: {
            'sans': 'Sans-serif',
            'serif': 'Serif',
            'mono': 'Monospace'
        }[x],
        key='font_style_radio'
    )

    st.session_state.text_style = st.radio(
        "Select Text Style:",
        ('normal', 'bold-headings', 'large-items'),
        format_func=lambda x: {
            'normal': 'Normal',
            'bold-headings': 'Bold Headings',
            'large-items': 'Large Items'
        }[x],
        key='text_style_radio'
    )

    st.subheader("AI Restaurant Details Lookup")
    if not st.session_state.ai_unlocked:
        st.text_input("Enter Password to Unlock AI:", type="password", key="ai_password_input", on_change=ai_password_submit_callback)
        if st.session_state.ai_error:
            st.error(st.session_state.ai_error)
        st.info("*Hint: `gemini`. For demo purposes only. Real API keys are securely handled via Streamlit Secrets.*")
    else:
        st.session_state.ai_location = st.text_input("Location (e.g., Rohini):", st.session_state.ai_location, key='ai_location_input')
        st.session_state.ai_meal_type = st.selectbox(
            "Meal Type:",
            ('', 'breakfast', 'lunch', 'dinner'),
            key='ai_meal_type_select'
        )
        st.button("Generate Restaurant Info with AI", on_click=fetch_ai_details_callback, disabled=st.session_state.ai_loading)
        if st.session_state.ai_loading:
            st.info(f"Searching for a restaurant in {st.session_state.ai_location or 'New Delhi'}...")
        if st.session_state.ai_error:
            st.error(st.session_state.ai_error)

    st.subheader("Restaurant Details")
    st.session_state.restaurant['name'] = st.text_input("Restaurant Name:", st.session_state.restaurant['name'], key='res_name')
    st.session_state.restaurant['address'] = st.text_input("Restaurant Address:", st.session_state.restaurant['address'], key='res_address')
    st.session_state.restaurant['city'] = st.text_input("City:", st.session_state.restaurant['city'], key='res_city')
    st.session_state.restaurant['state'] = st.text_input("State:", st.session_state.restaurant['state'], key='res_state')
    st.session_state.restaurant['telephone'] = st.text_input("Telephone:", st.session_state.restaurant['telephone'], key='res_tel')
    st.session_state.restaurant['gstin'] = st.text_input("GSTIN:", st.session_state.restaurant['gstin'], key='res_gstin')

    st.subheader("Order Specifics")
    st.session_state.order_info['orderNo'] = st.text_input("Order No.:", st.session_state.order_info['orderNo'], key='order_no')
    st.session_state.order_info['tableNo'] = st.text_input("Table No.:", st.session_state.order_info['tableNo'], key='table_no')
    st.session_state.order_info['serverName'] = st.text_input("Server Name:", st.session_state.order_info['serverName'], key='server_name')
    st.session_state.order_info['orderDateTime'] = st.date_input("Order Date:", value=datetime.datetime.fromisoformat(st.session_state.order_info['orderDateTime']), key='order_date_input')
    st.session_state.order_info['orderDateTime'] = st.time_input("Order Time:", value=datetime.time.fromisoformat(st.session_state.order_info['orderDateTime'].split('T')[1]), key='order_time_input')
    st.session_state.order_info['orderDateTime'] = datetime.datetime.combine(st.session_state.order_info['orderDateTime'].date(), st.session_state.order_info['orderDateTime'].time()).isoformat(timespec='minutes')

    st.subheader("Bill Options")
    st.session_state.currency_symbol = st.selectbox(
        "Currency Symbol:",
        ('₹', '$', '€', '£'),
        key='currency_symbol_select'
    )
    st.session_state.use_comma_decimal_separator = st.checkbox("Use Comma for Decimals (Ex: 40,50)", st.session_state.use_comma_decimal_separator, key='comma_decimal_checkbox')
    
    if st.session_state.bill_type != 'kot':
        st.session_state.include_service_charge = st.checkbox("Include Service Charge (10%)", st.session_state.include_service_charge, key='service_charge_checkbox')
        st.info("Note: Service charge is optional and not a mandatory tax in India.")
        st.session_state.gst_rate = st.number_input("GST Rate (%):", min_value=0, max_value=18, value=st.session_state.gst_rate, step=1, key='gst_rate_input')
        st.session_state.discount_percentage = st.number_input("Discount (%):", min_value=0.0, max_value=100.0, value=st.session_state.discount_percentage, step=0.5, key='discount_percentage_input')
        st.session_state.round_off_total = st.checkbox("Round Off Total Amount", st.session_state.round_off_total, key='round_off_checkbox')
        st.session_state.tip_amount = st.number_input(f"Tip Amount ({st.session_state.currency_symbol}):", min_value=0.0, value=st.session_state.tip_amount, step=0.01, key='tip_amount_input')
        
        st.subheader("Payment Details")
        st.session_state.credit_card_last_four = st.text_input("Credit Card (Last 4 Digits):", st.session_state.credit_card_last_four, max_chars=4, key='cc_last_four')
        st.session_state.payment_details['transactionType'] = st.text_input("Transaction Type:", st.session_state.payment_details['transactionType'], key='txn_type')
        st.session_state.payment_details['authorization'] = st.text_input("Authorization:", st.session_state.payment_details['authorization'], key='auth_code')
        st.session_state.payment_details['paymentCode'] = st.text_input("Payment Code:", st.session_state.payment_details['paymentCode'], key='pay_code')
        st.session_state.payment_details['paymentId'] = st.text_input("Payment ID:", st.session_state.payment_details['paymentId'], key='pay_id')
        st.session_state.payment_details['cardReader'] = st.text_input("Card Reader:", st.session_state.payment_details['cardReader'], key='card_reader')

    st.session_state.display_customer_copy = st.checkbox("Display 'Customer Copy'", st.session_state.display_customer_copy, key='cust_copy_checkbox')
    st.session_state.include_logo = st.checkbox("Include Logo Placeholder", st.session_state.include_logo, key='logo_checkbox')
    st.session_state.thank_you_message = st.text_area("Thank You Message:", st.session_state.thank_you_message, height=50, key='thank_you_msg')
    st.session_state.promotional_message = st.text_area("Promotional Message:", st.session_state.promotional_message, height=50, key='promo_msg')

    st.subheader("Actions")
    st.button("Reset Bill", on_click=reset_bill_callback)

# --- Item Management Section (Main Column) ---
st.header("Order Items")
col_header = st.columns([2, 1, 1, 2, 2])
with col_header[0]: st.markdown("Item Name")
with col_header[1]: st.markdown("Qty")
with col_header[2]: st.markdown(f"Price ({st.session_state.currency_symbol})")
with col_header[3]: st.markdown("Notes")
with col_header[4]: st.markdown("Actions") # For duplicate/move/remove buttons

for i, item in enumerate(st.session_state.items):
    cols = st.columns([2, 1, 1, 2, 0.5, 0.5, 1]) # Adjusted column widths for buttons
    
    with cols[0]:
        st.session_state.items[i]['name'] = st.text_input(
            "##", item['name'], key=f"item_name_{item['id']}", label_visibility="collapsed"
        )
    with cols[1]:
        st.session_state.items[i]['quantity'] = st.number_input(
            "##", item['quantity'], min_value=1, key=f"item_qty_{item['id']}", label_visibility="collapsed"
        )
    with cols[2]:
        st.session_state.items[i]['price'] = st.number_input(
            "##", item['price'], min_value=0.0, format="%.2f", disabled=(st.session_state.bill_type == 'kot'), key=f"item_price_{item['id']}", label_visibility="collapsed"
        )
    with cols[3]:
        st.session_state.items[i]['notes'] = st.text_input(
            "##", item['notes'], key=f"item_notes_{item['id']}", label_visibility="collapsed"
        )
    
    with cols[4]: # Duplicate button
        st.button("📋", on_click=duplicate_item_callback, args=(item['id'],), key=f"dup_item_{item['id']}", help="Duplicate Item")
    with cols[5]: # Up/Down buttons
        st.button("⬆️", on_click=move_item_callback, args=(item['id'], 'up'), key=f"move_up_{item['id']}", help="Move Up")
        st.button("⬇️", on_click=move_item_callback, args=(item['id'], 'down'), key=f"move_down_{item['id']}", help="Move Down")
    with cols[6]: # Remove button
        st.button("🗑️", on_click=remove_item_callback, args=(item['id'],), key=f"remove_item_{item['id']}", help="Remove Item")

st.button("Add Item", on_click=add_item_callback)

# --- Bill Preview (Main Column) ---
st.header("Bill Preview")

def generate_bill_html(
    restaurant, order_info, items, include_service_charge, service_charge_percentage,
    tip_amount, discount_percentage, round_off_total, gst_rate, bill_type, font_style,
    text_style, currency_symbol, use_comma_decimal_separator, credit_card_last_four,
    payment_details, display_customer_copy, thank_you_message, promotional_message,
    include_logo, for_download=False
):
    """Generates the HTML content for the bill preview."""
    
    # Calculate totals again for consistent display
    subtotal_before_discount = sum(item['quantity'] * item['price'] for item in items)
    discount_amount = (subtotal_before_discount * discount_percentage) / 100
    subtotal_after_discount = subtotal_before_discount - discount_amount

    service_charge = (subtotal_after_discount * service_charge_percentage) / 100 if include_service_charge else 0
    cgst = (subtotal_after_discount * (gst_rate / 2)) / 100
    sgst = (subtotal_after_discount * (gst_rate / 2)) / 100
    
    total_amount = subtotal_after_discount + service_charge + cgst + sgst + tip_amount
    if round_off_total:
        total_amount = round(total_amount)

    # Base CSS classes based on bill_type
    bill_classes = ""
    if bill_type == 'kot':
        bill_classes = "border border-dashed border-gray-400 p-4 shadow-inner max-w-sm mx-auto"
    elif bill_type == 'simple-receipt':
        bill_classes = "border border-gray-300 p-4 shadow-lg max-w-sm mx-auto"
    elif bill_type in ['thermal-1', 'thermal-2']:
        bill_classes = "border border-gray-300 p-3 shadow-lg max-w-xs mx-auto text-sm" # Smaller for thermal
    else: # customer-bill
        bill_classes = "rounded-lg shadow-xl p-6 max-w-lg mx-auto"
    
    # Font and text styles
    font_class = {'sans': 'font-sans', 'serif': 'font-serif', 'mono': 'font-mono'}.get(font_style, 'font-sans')
    
    # HTML content construction
    html_content = f"""
    <style>
        .bill-container {{
            background-color: {'#f9fafb' if bill_type == 'customer-bill' else '#ffffff'};
            color: {'#1f2937' if bill_type == 'customer-bill' else '#111827'};
            {font_class};
            border-radius: {'0.5rem' if bill_type == 'customer-bill' else '0'};
            padding: {'1.5rem' if bill_type == 'customer-bill' else '1rem'};
            box-shadow: {'0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)' if bill_type == 'customer-bill' else 'none'};
            width: 100%;
            overflow-x: auto; /* Ensure horizontal scroll for smaller screens */
            box-sizing: border-box;
        }}
        .text-center {{ text-align: center; }}
        .mb-2 {{ margin-bottom: 0.5rem; }}
        .mb-4 {{ margin-bottom: 1rem; }}
        .mb-6 {{ margin-bottom: 1.5rem; }}
        .mt-1 {{ margin-top: 0.25rem; }}
        .mt-2 {{ margin-top: 0.5rem; }}
        .mt-4 {{ margin-top: 1rem; }}
        .mt-6 {{ margin-top: 1.5rem; }}
        .mt-8 {{ margin-top: 2rem; }}
        .flex {{ display: flex; }}
        .justify-between {{ justify-content: space-between; }}
        .items-start {{ align-items: flex-start; }}
        .text-xs {{ font-size: 0.75rem; }}
        .text-sm {{ font-size: 0.875rem; }}
        .text-md {{ font-size: 1rem; }}
        .text-lg {{ font-size: 1.125rem; }}
        .text-xl {{ font-size: 1.25rem; }}
        .text-2xl {{ font-size: 1.5rem; }}
        .text-3xl {{ font-size: 1.875rem; }}
        .font-bold {{ font-weight: 700; }}
        .font-semibold {{ font-weight: 600; }}
        .uppercase {{ text-transform: uppercase; }}
        .italic {{ font-style: italic; }}
        .border-b {{ border-bottom-width: 1px; border-color: #e5e7eb; }}
        .pb-2 {{ padding-bottom: 0.5rem; }}
        .pt-2 {{ padding-top: 0.5rem; }}
        .border-t {{ border-top-width: 1px; border-color: #e5e7eb; }}
        .py-1 {{ padding-top: 0.25rem; padding-bottom: 0.25rem; }}
        .py-2 {{ padding-top: 0.5rem; padding-bottom: 0.5rem; }}
        .px-3 {{ padding-left: 0.75rem; padding-right: 0.75rem; }}
        .bg-blue-100 {{ background-color: #dbeafe; }}
        .rounded-md {{ border-radius: 0.375rem; }}
        .px-3 {{ padding-left: 0.75rem; padding-right: 0.75rem; }}
        .text-blue-800 {{ color: #1e40af; }}
        .text-red-600 {{ color: #dc2626; }}
        .text-gray-600 {{ color: #4b5563; }}
        .text-gray-700 {{ color: #374151; }}
        .text-gray-800 {{ color: #1f2937; }}
        .table-auto {{ width: 100%; border-collapse: collapse; }}
        .table-auto th, .table-auto td {{ padding: 0.5rem 0.5rem; text-align: left; }}
        .whitespace-nowrap {{ white-space: nowrap; }}
        .rounded-tl-md {{ border-top-left-radius: 0.375rem; }}
        .rounded-tr-md {{ border-top-right-radius: 0.375rem; }}
        .bg-gray-200 {{ background-color: #e5e7eb; }}
        .w-24 {{ width: 6rem; }}
        .h-24 {{ height: 6rem; }}
        .bg-gray-300 {{ background-color: #d1d5db; }}
        .flex-row {{ flex-direction: row; }}
        .items-center {{ align-items: center; }}
        .justify-center {{ justify-content: center; }}
        .mx-auto {{ margin-left: auto; margin-right: auto; }}
        
        .font-sans {{ font-family: 'Inter', sans-serif; }}
        .font-serif {{ font-family: 'Playfair Display', serif; }}
        .font-mono {{ font-family: 'Roboto Mono', monospace; }}

        .bill-header {{
            font-size: {'1.875rem' if bill_type == 'customer-bill' else '1.5rem' if bill_type == 'simple-receipt' else '1.25rem'};
            font-weight: 700;
            text-align: center;
            margin-bottom: 1rem;
            {
                'text-transform: uppercase;' if text_style == 'bold-headings' else ''
            }
        }}
        .item-row td {{
            font-size: {'1.125rem' if text_style == 'large-items' else '0.875rem'};
            line-height: 1.5;
        }}
        .bill-summary-value {{
            font-weight: {'bold' if text_style == 'bold-headings' else 'normal'};
        }}
        @media print {{
            .bill-container {{
                box-shadow: none;
                border: none;
                padding: 0;
            }}
        }}
    </style>
    <div class="bill-container">
    """
    
    if display_customer_copy and bill_type != 'kot':
        html_content += """<p class="text-center text-sm font-bold mb-2 border-b pb-2 border-gray-300">--- CUSTOMER COPY ---</p>"""

    if include_logo and (bill_type == 'customer-bill' or bill_type == 'simple-receipt' or bill_type == 'thermal-2'):
        html_content += """
        <div class="text-center mb-4">
            <div style="margin-left: auto; margin-right: auto; width: 6rem; height: 6rem; background-color: #e5e7eb; border-radius: 9999px; display: flex; align-items: center; justify-content: center; color: #6b7280; margin-bottom: 0.5rem;">
                Your Logo
            </div>
        </div>
        """

    header_text = 'KITCHEN ORDER TICKET' if bill_type == 'kot' else restaurant['name']
    html_content += f"""<h2 class="bill-header">{header_text}</h2>"""

    if bill_type != 'kot':
        html_content += f"""
        <div class="text-center mb-2">
            <p class="text-xs text-gray-600">{restaurant['address']}, {restaurant['city']}, {restaurant['state']}</p>
            <p class="text-xs text-gray-600">Tel: {restaurant['telephone']}</p>
            <p class="text-xs text-gray-600">GSTIN: {restaurant['gstin']}</p>
        </div>
        """
    
    # Order Info Section
    if bill_type in ['thermal-1', 'thermal-2']:
        html_content += f"""
        <div class="text-center text-sm font-mono mb-2">
            <p>ORDER: {order_info['orderNo']}</p>
            <p>HOST: {order_info['serverName'].upper()}</p>
            <div class="flex justify-between items-center text-xs mt-1">
                <span>{datetime.datetime.fromisoformat(order_info['orderDateTime']).strftime('%d/%m/%Y')}</span>
                <span>{datetime.datetime.fromisoformat(order_info['orderDateTime']).strftime('%I:%M%p').upper().replace(' ', '')}</span>
            </div>
        </div>
        """
    else: # General Order Info
        html_content += f"""
        <div class="flex justify-between items-start mb-4 text-xs">
            <div>
                <p><strong>Order No:</strong> {order_info['orderNo']}</p>
                <p><strong>Table No:</strong> {order_info['tableNo']}</p>
            </div>
            <div style="text-align: right;">
                <p><strong>Date:</strong> {order_info['orderDateTime'].split('T')[0]}</p>
                <p><strong>Time:</strong> {order_info['orderDateTime'].split('T')[1]}</p>
                <p><strong>Server:</strong> {order_info['serverName']}</p>
            </div>
        </div>
        """
    
    if bill_type == 'thermal-2':
        html_content += f"""<div class="text-center text-sm mb-4">Table - {order_info['tableNo']}</div>"""

    # Items Table
    html_content += f"""
    <table class="table-auto" style="border-top: 1px dashed #e5e7eb; border-bottom: 1px dashed #e5e7eb; margin-bottom: 1rem;">
        <thead>
            <tr>
                <th style="padding: 0.5rem 0.5rem; text-align: left; font-size: {'0.875rem' if bill_type in ['thermal-1', 'thermal-2'] else '1rem'};">Item</th>
                <th style="padding: 0.5rem 0.5rem; text-align: left; font-size: {'0.875rem' if bill_type in ['thermal-1', 'thermal-2'] else '1rem'};">Qty</th>
                {'' if bill_type == 'kot' else f'<th style="padding: 0.5rem 0.5rem; text-align: left; font-size: {"0.875rem" if bill_type in ["thermal-1", "thermal-2"] else "1rem"};">Price ({currency_symbol})</th>'}
                {'' if bill_type == 'kot' else f'<th style="padding: 0.5rem 0.5rem; text-align: right; font-size: {"0.875rem" if bill_type in ["thermal-1", "thermal-2"] else "1rem"};">Total ({currency_symbol})</th>'}
                {'' if bill_type != 'kot' else f'<th style="padding: 0.5rem 0.5rem; text-align: left; font-size: {"0.875rem" if bill_type in ["thermal-1", "thermal-2"] else "1rem"};">Notes</th>'}
            </tr>
        </thead>
        <tbody>
    """
    for item in items:
        item_total = item['quantity'] * item['price']
        html_content += f"""
            <tr class="item-row">
                <td style="padding: 0.25rem 0.5rem; white-space: nowrap; font-size: {'0.75rem' if bill_type in ["thermal-1", "thermal-2"] else ("1.125rem" if text_style == "large-items" else "0.875rem")};">{item['name']}</td>
                <td style="padding: 0.25rem 0.5rem; white-space: nowrap; font-size: {'0.75rem' if bill_type in ["thermal-1", "thermal-2"] else ("1.125rem" if text_style == "large-items" else "0.875rem")};">{item['quantity']}</td>
                {'' if bill_type == 'kot' else f'<td style="padding: 0.25rem 0.5rem; white-space: nowrap; font-size: {"0.75rem" if bill_type in ["thermal-1", "thermal-2"] else ("1.125rem" if text_style == "large-items" else "0.875rem")};">{format_number(item["price"], use_comma_decimal_separator)}</td>'}
                {'' if bill_type == 'kot' else f'<td style="padding: 0.25rem 0.5rem; white-space: nowrap; text-align: right; font-size: {"0.75rem" if bill_type in ["thermal-1", "thermal-2"] else ("1.125rem" if text_style == "large-items" else "0.875rem")};">{format_number(item_total, use_comma_decimal_separator)}</td>'}
                {'' if bill_type != 'kot' else f'<td style="padding: 0.25rem 0.5rem; white-space: nowrap; font-size: {"0.75rem" if bill_type in ["thermal-1", "thermal-2"] else ("1.125rem" if text_style == "large-items" else "0.875rem")}; color: #6b7280;">{item["notes"]}</td>'}
            </tr>
        """
    html_content += """
        </tbody>
    </table>
    """

    # Summary Section
    if bill_type != 'kot':
        html_content += f"""
        <div style="display: flex; justify-content: flex-end; width: 100%;">
            <div style="width: {'100%' if bill_type in ['thermal-1', 'simple-receipt', 'thermal-2'] else '50%'};">
                <div style="display: flex; justify-content: space-between; padding: 0.25rem 0; border-bottom: 1px solid #e5e7eb;">
                    <span class="text-sm text-gray-700">Subtotal:</span>
                    <span class="text-sm font-medium text-gray-800 bill-summary-value">{currency_symbol} {format_number(subtotal_before_discount, use_comma_decimal_separator)}</span>
                </div>
                {'' if discount_percentage == 0 else f'''
                <div style="display: flex; justify-content: space-between; padding: 0.25rem 0; border-bottom: 1px solid #e5e7eb;">
                    <span class="text-sm text-gray-700">Discount ({discount_percentage}%):</span>
                    <span class="text-sm font-medium" style="color: #dc2626;">- {currency_symbol} {format_number(discount_amount, use_comma_decimal_separator)}</span>
                </div>
                '''}
                {'' if not include_service_charge else f'''
                <div style="display: flex; justify-content: space-between; padding: 0.25rem 0; border-bottom: 1px solid #e5e7eb;">
                    <span class="text-sm text-gray-700">Service Charge ({service_charge_percentage}%):</span>
                    <span class="text-sm font-medium text-gray-800 bill-summary-value">{currency_symbol} {format_number(service_charge, use_comma_decimal_separator)}</span>
                </div>
                '''}
                <div style="display: flex; justify-content: space-between; padding: 0.25rem 0; border-bottom: 1px solid #e5e7eb;">
                    <span class="text-sm text-gray-700">CGST ({gst_rate / 2}%):</span>
                    <span class="text-sm font-medium text-gray-800 bill-summary-value">{currency_symbol} {format_number(cgst, use_comma_decimal_separator)}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.25rem 0; border-bottom: 1px solid #e5e7eb;">
                    <span class="text-sm text-gray-700">SGST ({gst_rate / 2}%):</span>
                    <span class="text-sm font-medium text-gray-800 bill-summary-value">{currency_symbol} {format_number(sgst, use_comma_decimal_separator)}</span>
                </div>
                {'' if tip_amount == 0 else f'''
                <div style="display: flex; justify-content: space-between; padding: 0.25rem 0; border-bottom: 1px solid #e5e7eb;">
                    <span class="text-sm text-gray-700">Tip:</span>
                    <span class="text-sm font-medium text-gray-800 bill-summary-value">{currency_symbol} {format_number(tip_amount, use_comma_decimal_separator)}</span>
                </div>
                '''}
                {'' if not round_off_total else f'''
                <div style="display: flex; justify-content: space-between; padding: 0.25rem 0; border-bottom: 1px solid #e5e7eb;">
                    <span class="text-sm text-gray-700">Round Off:</span>
                    <span class="text-sm font-medium text-gray-800 bill-summary-value">{currency_symbol} {format_number(total_amount - (subtotal_after_discount + service_charge + cgst + sgst + tip_amount), use_comma_decimal_separator)}</span>
                </div>
                '''}
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0.75rem; margin-top: 0.5rem; background-color: #dbeafe; border-radius: 0.375rem;">
                    <span class="text-md font-semibold" style="color: #1e40af;">Total Amount:</span>
                    <span class="text-md font-bold" style="color: #1e40af;">{currency_symbol} {format_number(total_amount, use_comma_decimal_separator)}</span>
                </div>

                {'' if not (credit_card_last_four or payment_details['transactionType']) else f'''
                <div style="margin-top: 1rem; border-top: 1px solid #e5e7eb; padding-top: 0.5rem; font-size: 0.75rem; color: #4b5563;">
                    <p><strong>{payment_details['transactionType']}</strong></p>
                    {'' if not credit_card_last_four else f'<p>VISA Credit XXXXXXXXXXXX{credit_card_last_four}</p>'}
                    {'' if not payment_details['authorization'] else f'<p>AUTHORIZATION: {payment_details["authorization"]}</p>'}
                    {'' if not payment_details['paymentCode'] else f'<p>PAYMENT CODE: {payment_details["paymentCode"]}</p>'}
                    {'' if not payment_details['paymentId'] else f'<p>PAYMENT ID: {payment_details["paymentId"]}</p>'}
                    {'' if not payment_details['cardReader'] else f'<p>CARD READER: {payment_details["cardReader"]}</p>'}
                </div>
                '''}
            </div>
        </div>
        """

    if bill_type in ['thermal-1', 'thermal-2']:
        html_content += """
        <div class="text-center mt-6">
            <div style="width: 100%; height: 2rem; background-color: #d1d5db; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; color: #4b5563; border-radius: 0.25rem;">
                BARCODE PLACEHOLDER
            </div>
        </div>
        """

    if promotional_message and (bill_type == 'customer-bill' or bill_type == 'simple-receipt' or bill_type == 'thermal-2'):
        html_content += f"""
        <div class="text-center mt-6 italic" style="color: #4b5563; font-size: 0.875rem; border-top: 1px solid #e5e7eb; padding-top: 1rem;">
            <p>{promotional_message}</p>
        </div>
        """
    
    final_thank_you_message = "--- END OF ORDER ---" if bill_type == 'kot' else thank_you_message
    final_thank_you_class = "font-bold text-lg mt-4" if bill_type == 'kot' else "text-sm"

    html_content += f"""
    <div class="text-center mt-6" style="color: #4b5563;">
        <p class="{final_thank_you_class}">{final_thank_you_message}</p>
    </div>
    </div>
    """
    
    if for_download: # Return plain text for download
        # This is a simplified way to get plain text, might not be perfectly formatted
        # for complex HTML, but works for structured content.
        # A better approach for plain text would be to build a text string directly.
        plain_text = f"""
{'-' * 40}
{'--- CUSTOMER COPY ---' if display_customer_copy and bill_type != 'kot' else ''}
{'--- KITCHEN ORDER TICKET ---' if bill_type == 'kot' else restaurant['name']}
{restaurant['address']}, {restaurant['city']}, {restaurant['state']}
Tel: {restaurant['telephone']}
GSTIN: {restaurant['gstin']}
{'-' * 40}

Order No: {order_info['orderNo']}
Table No: {order_info['tableNo']}
Date: {order_info['orderDateTime'].split('T')[0]}
Time: {order_info['orderDateTime'].split('T')[1]}
Server: {order_info['serverName']}

{'-' * 40}
Item             Qty   Price     Total
{'-' * 40}
"""
        for item in items:
            item_total_formatted = format_number(item['quantity'] * item['price'], use_comma_decimal_separator) if bill_type != 'kot' else ''
            item_price_formatted = format_number(item['price'], use_comma_decimal_separator) if bill_type != 'kot' else ''
            item_notes = f" ({item['notes']})" if item['notes'] and bill_type == 'kot' else ''
            plain_text += f"{item['name'][:15]:<15} {item['quantity']:<5} {item_price_formatted:<9} {item_total_formatted:>9}{item_notes}\n"
        
        if bill_type != 'kot':
            plain_text += f"""
{'-' * 40}
Subtotal: {currency_symbol} {format_number(subtotal_before_discount, use_comma_decimal_separator)}
"""
            if discount_percentage > 0:
                plain_text += f"Discount ({discount_percentage}%): - {currency_symbol} {format_number(discount_amount, use_comma_decimal_separator)}\n"
            if include_service_charge:
                plain_text += f"Service Charge ({service_charge_percentage}%): {currency_symbol} {format_number(service_charge, use_comma_decimal_separator)}\n"
            plain_text += f"CGST ({gst_rate / 2}%): {currency_symbol} {format_number(cgst, use_comma_decimal_separator)}\n"
            plain_text += f"SGST ({gst_rate / 2}%): {currency_symbol} {format_number(sgst, use_comma_decimal_separator)}\n"
            if tip_amount > 0:
                plain_text += f"Tip: {currency_symbol} {format_number(tip_amount, use_comma_decimal_separator)}\n"
            if round_off_total:
                plain_text += f"Round Off: {currency_symbol} {format_number(total_amount - (subtotal_after_discount + service_charge + cgst + sgst + tip_amount), use_comma_decimal_separator)}\n"
            
            plain_text += f"""
{'-' * 40}
TOTAL AMOUNT: {currency_symbol} {format_number(total_amount, use_comma_decimal_separator)}
{'-' * 40}
"""
            if credit_card_last_four or payment_details['transactionType']:
                plain_text += f"\nPAYMENT INFO:\n"
                plain_text += f"  Transaction Type: {payment_details['transactionType']}\n"
                if credit_card_last_four:
                    plain_text += f"  VISA Credit XXXXXXXXXXXX{credit_card_last_four}\n"
                if payment_details['authorization']:
                    plain_text += f"  AUTH: {payment_details['authorization']}\n"
                if payment_details['paymentCode']:
                    plain_text += f"  PAY CODE: {payment_details['paymentCode']}\n"
                if payment_details['paymentId']:
                    plain_text += f"  PAY ID: {payment_details['paymentId']}\n"
                if payment_details['cardReader']:
                    plain_text += f"  READER: {payment_details['cardReader']}\n"
                plain_text += f"{'-' * 40}\n"

        if promotional_message and (bill_type == 'customer-bill' or bill_type == 'simple-receipt' or bill_type == 'thermal-2'):
            plain_text += f"\n{promotional_message}\n"
            plain_text += f"{'-' * 40}\n"
        
        plain_text += f"\n{final_thank_you_message}\n"
        plain_text += f"{'-' * 40}\n"
        return plain_text
    
    return html_content

# Render the bill preview
st.markdown(generate_bill_html(
    st.session_state.restaurant,
    st.session_state.order_info,
    st.session_state.items,
    st.session_state.include_service_charge,
    st.session_state.service_charge_percentage,
    st.session_state.tip_amount,
    st.session_state.discount_percentage,
    st.session_state.round_off_total,
    st.session_state.gst_rate,
    st.session_state.bill_type,
    st.session_state.font_style,
    st.session_state.text_style,
    st.session_state.currency_symbol,
    st.session_state.use_comma_decimal_separator,
    st.session_state.credit_card_last_four,
    st.session_state.payment_details,
    st.session_state.display_customer_copy,
    st.session_state.thank_you_message,
    st.session_state.promotional_message,
    st.session_state.include_logo
), unsafe_allow_html=True)

# Action buttons for download/copy (placed after preview for better UX)
st.button("Download Bill Text", on_click=download_bill_text_callback)
st.info("For 'Copy Bill Text', please select the content in the preview above and copy manually, or use the 'Download Bill Text' option.")
