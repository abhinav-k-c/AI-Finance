import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import io

def validate_financial_data(df):
    """Validate the uploaded financial data"""
    required_columns = ['date', 'description', 'amount', 'category', 'type']
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns.str.lower()]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Check for empty values in critical columns
    if df[['amount', 'type']].isnull().any().any():
        return False, "Amount and Type columns cannot have empty values"
    
    # Validate amount column (should be numeric)
    try:
        df['amount'] = pd.to_numeric(df['amount'])
    except:
        return False, "Amount column contains non-numeric values"
    
    # Validate type column (should be income or expense)
    valid_types = ['income', 'expense']
    if not df['type'].str.lower().isin(valid_types).all():
        return False, "Type column should only contain 'income' or 'expense'"
    
    return True, "Data validation successful"

def manual_data_entry():
    """Manual data entry form"""
    st.subheader("📝 Manual Data Entry")
    
    with st.form("manual_entry_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            entry_date = st.date_input("Date", value=date.today())
            description = st.text_input("Description", placeholder="e.g., Grocery shopping")
            amount = st.number_input("Amount", min_value=0.01, step=0.01)
        
        with col2:
            category = st.selectbox("Category", [
                "Food & Dining", "Transportation", "Shopping", "Entertainment",
                "Bills & Utilities", "Healthcare", "Education", "Travel",
                "Salary", "Business", "Investments", "Other"
            ])
            entry_type = st.selectbox("Type", ["Expense", "Income"])
        
        submitted = st.form_submit_button("Add Entry")
        
        if submitted:
            if description and amount > 0:
                # Initialize session state for storing entries
                if 'manual_entries' not in st.session_state:
                    st.session_state.manual_entries = []
                
                # Add new entry
                new_entry = {
                    'date': entry_date.strftime('%Y-%m-%d'),
                    'description': description,
                    'amount': amount,
                    'category': category,
                    'type': entry_type.lower()
                }
                
                st.session_state.manual_entries.append(new_entry)
                st.success("✅ Entry added successfully!")
                st.rerun()
            else:
                st.error("Please fill in all required fields")

def display_manual_entries():
    """Display manually entered data"""
    if 'manual_entries' in st.session_state and st.session_state.manual_entries:
        st.subheader("📊 Your Manual Entries")
        
        df = pd.DataFrame(st.session_state.manual_entries)
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            total_income = df[df['type'] == 'income']['amount'].sum()
            st.metric("Total Income", f"₹{total_income:,.2f}")
        with col2:
            total_expense = df[df['type'] == 'expense']['amount'].sum()
            st.metric("Total Expenses", f"₹{total_expense:,.2f}")
        with col3:
            balance = total_income - total_expense
            st.metric("Balance", f"₹{balance:,.2f}", delta=f"₹{balance:,.2f}")
        
        # Display data table
        st.dataframe(
            df.sort_values('date', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Option to clear all entries
        if st.button("🗑️ Clear All Manual Entries", type="secondary"):
            st.session_state.manual_entries = []
            st.success("All manual entries cleared!")
            st.rerun()
        
        # Option to download manual entries as CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Manual Entries as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"manual_entries_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def csv_upload_section():
    """CSV file upload section"""
    st.subheader("📂 CSV File Upload")
    
    # Sample CSV format
    with st.expander("📋 View Sample CSV Format"):
        sample_data = {
            'date': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'description': ['Salary Credit', 'Grocery Shopping', 'Electric Bill'],
            'amount': [50000, 2500, 1200],
            'category': ['Salary', 'Food & Dining', 'Bills & Utilities'],
            'type': ['income', 'expense', 'expense']
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True, hide_index=True)
        
        # Download sample CSV
        csv_sample = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=csv_sample,
            file_name="sample_financial_data.csv",
            mime="text/csv"
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your financial data in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} records.")
            
            # Display basic info about the uploaded data
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Rows:** {len(df)}")
                st.info(f"**Columns:** {len(df.columns)}")
            with col2:
                st.info(f"**File Size:** {uploaded_file.size} bytes")
                st.info(f"**File Name:** {uploaded_file.name}")
            
            # Show column names
            st.write("**Column Names Found:**")
            st.write(", ".join(df.columns.tolist()))
            
            # Data validation
            is_valid, message = validate_financial_data(df.copy())
            
            if is_valid:
                st.success(f"✅ {message}")
                
                # Preview the data
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Data summary
                if 'amount' in df.columns.str.lower():
                    df_normalized = df.copy()
                    df_normalized.columns = df_normalized.columns.str.lower()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_records = len(df_normalized)
                        st.metric("Total Records", total_records)
                    with col2:
                        if 'type' in df_normalized.columns:
                            income_records = len(df_normalized[df_normalized['type'].str.lower() == 'income'])
                            st.metric("Income Records", income_records)
                    with col3:
                        if 'type' in df_normalized.columns:
                            expense_records = len(df_normalized[df_normalized['type'].str.lower() == 'expense'])
                            st.metric("Expense Records", expense_records)
                
                # Store in session state
                if st.button("💾 Load This Data", type="primary"):
                    st.session_state.uploaded_data = df
                    st.success("✅ Data loaded successfully! You can now use this data in other pages.")
                    
            else:
                st.error(f"❌ Data Validation Failed: {message}")
                st.write("**Please ensure your CSV file has the following structure:**")
                st.write("- **date**: Date of transaction (YYYY-MM-DD format)")
                st.write("- **description**: Description of the transaction")
                st.write("- **amount**: Transaction amount (numeric)")
                st.write("- **category**: Category of transaction")
                st.write("- **type**: 'income' or 'expense'")
                
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.write("Please make sure your file is a valid CSV format.")

def data_management_section():
    """Data management and preview section"""
    st.subheader("🔍 Data Management")
    
    # Check if we have any data
    has_manual_data = 'manual_entries' in st.session_state and st.session_state.manual_entries
    has_uploaded_data = 'uploaded_data' in st.session_state
    
    if not has_manual_data and not has_uploaded_data:
        st.info("📝 No data available yet. Please add data using manual entry or CSV upload above.")
        return
    
    # Data source selection
    data_sources = []
    if has_manual_data:
        data_sources.append("Manual Entries")
    if has_uploaded_data:
        data_sources.append("Uploaded CSV")
    
    if len(data_sources) > 1:
        selected_source = st.selectbox("Select Data Source to View", data_sources)
    else:
        selected_source = data_sources[0]
    
    # Display selected data
    if selected_source == "Manual Entries" and has_manual_data:
        df = pd.DataFrame(st.session_state.manual_entries)
    elif selected_source == "Uploaded CSV" and has_uploaded_data:
        df = st.session_state.uploaded_data
    else:
        return
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        if 'amount' in df.columns:
            avg_amount = df['amount'].mean()
            st.metric("Avg Amount", f"₹{avg_amount:,.2f}")
    with col3:
        if 'date' in df.columns:
            date_range = f"{df['date'].min()} to {df['date'].max()}"
            st.metric("Date Range", date_range)
    with col4:
        if 'category' in df.columns:
            unique_categories = df['category'].nunique()
            st.metric("Categories", unique_categories)
    
    # Display full data
    st.dataframe(df, use_container_width=True, hide_index=True)

def main():
    """Main function for the Data Input & Management page"""
    
    st.title("💼 Financial Data Input & Management")
    st.markdown("---")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📝 Manual Entry", "📂 CSV Upload", "🔍 Data Management"])
    
    with tab1:
        manual_data_entry()
        st.markdown("---")
        display_manual_entries()
    
    with tab2:
        csv_upload_section()
    
    with tab3:
        data_management_section()

if __name__ == "__main__":
    main()
