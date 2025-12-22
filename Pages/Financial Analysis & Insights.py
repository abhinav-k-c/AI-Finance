import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import calendar

def get_combined_data():
    """Get combined data from manual entries and uploaded CSV"""
    combined_df = pd.DataFrame()
    
    # Get manual entries
    if 'manual_entries' in st.session_state and st.session_state.manual_entries:
        manual_df = pd.DataFrame(st.session_state.manual_entries)
        combined_df = pd.concat([combined_df, manual_df], ignore_index=True)
    
    # Get uploaded data
    if 'uploaded_data' in st.session_state:
        uploaded_df = st.session_state.uploaded_data.copy()
        # Normalize column names
        uploaded_df.columns = uploaded_df.columns.str.lower()
        combined_df = pd.concat([combined_df, uploaded_df], ignore_index=True)
    
    if not combined_df.empty:
        # Ensure data types
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df['amount'] = pd.to_numeric(combined_df['amount'])
        combined_df['type'] = combined_df['type'].str.lower()
        
        # Sort by date
        combined_df = combined_df.sort_values('date')
        
    return combined_df

def financial_overview_dashboard():
    """Create financial overview dashboard"""
    st.subheader("📊 Financial Overview Dashboard")
    
    df = get_combined_data()
    
    if df.empty:
        st.warning("⚠️ No financial data available. Please add data in the 'Data Input & Management' page first.")
        return
    
    # Calculate key metrics
    total_income = df[df['type'] == 'income']['amount'].sum()
    total_expenses = df[df['type'] == 'expense']['amount'].sum()
    net_balance = total_income - total_expenses
    savings_rate = (net_balance / total_income * 100) if total_income > 0 else 0
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Income",
            value=f"₹{total_income:,.2f}",
            delta=f"₹{total_income:,.2f}"
        )
    
    with col2:
        st.metric(
            label="💸 Total Expenses", 
            value=f"₹{total_expenses:,.2f}",
            delta=f"-₹{total_expenses:,.2f}",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="💵 Net Balance",
            value=f"₹{net_balance:,.2f}",
            delta=f"₹{net_balance:,.2f}",
            delta_color="normal" if net_balance >= 0 else "inverse"
        )
    
    with col4:
        st.metric(
            label="📈 Savings Rate",
            value=f"{savings_rate:.1f}%",
            delta=f"{savings_rate:.1f}%"
        )

def spending_analysis_alerts():
    """Analyze spending patterns and generate smart alerts"""
    st.subheader("🚨 Smart Spending Alerts")
    
    df = get_combined_data()
    
    if df.empty:
        st.info("📝 No financial data available for analysis.")
        return
    
    # Filter expense data
    expense_df = df[df['type'] == 'expense'].copy()
    
    if expense_df.empty:
        st.info("📝 No expense data available for analysis.")
        return
    
    # Calculate monthly averages
    expense_df['month'] = expense_df['date'].dt.to_period('M')
    monthly_expenses = expense_df.groupby('month')['amount'].sum()
    
    if len(monthly_expenses) == 0:
        st.info("📝 No expense data available for analysis.")
        return
    
    avg_monthly_expense = monthly_expenses.mean()
    current_month = expense_df['date'].dt.to_period('M').max()
    current_month_expense = monthly_expenses.get(current_month, 0)
    
    # Category-wise analysis
    category_expenses = expense_df.groupby('category')['amount'].sum().sort_values(ascending=False)
    
    alerts = []
    
    # Check if current month spending is above average
    if len(monthly_expenses) > 1 and current_month_expense > avg_monthly_expense * 1.2:
        overspend_amount = current_month_expense - avg_monthly_expense
        alerts.append({
            'type': 'warning',
            'message': f"⚠️ You've overspent ₹{overspend_amount:,.2f} this month compared to your average!"
        })
    
    # Check top spending categories
    if len(category_expenses) > 0:
        top_category = category_expenses.index[0]
        top_amount = category_expenses.iloc[0]  # Access the first value
        total_expenses = category_expenses.sum()
        
        # Ensure we have valid numeric values
        if pd.notna(top_amount) and pd.notna(total_expenses) and total_expenses > 0:
            st.text(top_amount)
            st.text(total_expenses)
            percentage = (top_amount / total_expenses) * 100
            
            if percentage > 40:
                alerts.append({
                    'type': 'info',
                    'message': f"💡 {top_category} accounts for {percentage:.1f}% of your total expenses (₹{top_amount:,.2f})"
                })
    
    # Check for high single transactions
    if len(expense_df) > 0:
        high_transactions = expense_df.nlargest(3, 'amount')
        if not high_transactions.empty:
            highest_expense = high_transactions.iloc[0]
            if highest_expense['amount'] > avg_monthly_expense * 0.3:
                alerts.append({
                    'type': 'info',
                    'message': f"💳 Large expense alert: ₹{highest_expense['amount']:,.2f} on {highest_expense['description']} ({highest_expense['date'].strftime('%Y-%m-%d')})"
                })
    
    # Display alerts
    if alerts:
        for alert in alerts:
            if alert['type'] == 'warning':
                st.warning(alert['message'])
            else:
                st.info(alert['message'])
    else:
        st.success("✅ Your spending patterns look healthy!")

def create_expense_income_chart():
    """Create expense vs income comparison chart"""
    df = get_combined_data()
    
    if df.empty:
        return None
    
    # Monthly comparison
    df['month'] = df['date'].dt.to_period('M').astype(str)
    monthly_data = df.groupby(['month', 'type'])['amount'].sum().reset_index()
    
    if monthly_data.empty:
        return None
    
    fig = px.bar(
        monthly_data,
        x='month',
        y='amount',
        color='type',
        title='📊 Monthly Income vs Expenses',
        color_discrete_map={'income': '#2E8B57', 'expense': '#DC143C'},
        barmode='group'
    )
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Amount (₹)",
        legend_title="Type",
        height=400
    )
    
    return fig

def create_category_pie_chart():
    """Create pie chart for expense categories"""
    df = get_combined_data()
    
    if df.empty:
        return None
    
    expense_categories = df[df['type'] == 'expense'].groupby('category')['amount'].sum().reset_index()
    
    if expense_categories.empty:
        return None
    
    fig = px.pie(
        expense_categories,
        values='amount',
        names='category',
        title='🥧 Expense Distribution by Category'
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def create_spending_trend_chart():
    """Create spending trend over time"""
    df = get_combined_data()
    
    if df.empty:
        return None
    
    # Daily cumulative spending
    daily_expenses = df[df['type'] == 'expense'].groupby(df['date'].dt.date)['amount'].sum().reset_index()
    daily_expenses['cumulative'] = daily_expenses['amount'].cumsum()
    
    if daily_expenses.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('📈 Daily Expenses', '📊 Cumulative Spending'),
        vertical_spacing=0.1
    )
    
    # Daily expenses line chart
    fig.add_trace(
        go.Scatter(
            x=daily_expenses['date'],
            y=daily_expenses['amount'],
            mode='lines+markers',
            name='Daily Expenses',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Cumulative spending
    fig.add_trace(
        go.Scatter(
            x=daily_expenses['date'],
            y=daily_expenses['cumulative'],
            mode='lines+markers',
            name='Cumulative',
            line=dict(color='blue', width=2),
            fill='tonexty'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="💹 Spending Trends Over Time"
    )
    
    return fig

def category_analysis():
    """Detailed category-wise analysis"""
    st.subheader("📈 Category-wise Analysis")
    
    df = get_combined_data()
    
    if df.empty:
        return
    
    # Category summary table
    expense_summary = df[df['type'] == 'expense'].groupby('category').agg({
        'amount': ['sum', 'mean', 'count'],
        'date': ['min', 'max']
    }).round(2)
    
    expense_summary.columns = ['Total Spent', 'Average Amount', 'Transactions', 'First Transaction', 'Last Transaction']
    expense_summary = expense_summary.sort_values('Total Spent', ascending=False)
    
    st.dataframe(expense_summary, use_container_width=True)
    
    # Top spending categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🏆 Top 5 Expense Categories**")
        top_categories = expense_summary.head().reset_index()
        for i, row in top_categories.iterrows():
            percentage = (row['Total Spent'] / expense_summary['Total Spent'].sum()) * 100
            st.write(f"{i+1}. **{row['category']}**: ₹{row['Total Spent']:,.2f} ({percentage:.1f}%)")
    
    with col2:
        st.write("**💡 Spending Insights**")
        total_transactions = expense_summary['Transactions'].sum()
        avg_transaction = expense_summary['Total Spent'].sum() / total_transactions
        most_frequent_category = expense_summary.loc[expense_summary['Transactions'].idxmax()].name
        
        st.write(f"• **Average transaction**: ₹{avg_transaction:.2f}")
        st.write(f"• **Most frequent category**: {most_frequent_category}")
        st.write(f"• **Total categories**: {len(expense_summary)}")
        st.write(f"• **Total transactions**: {int(total_transactions)}")

def monthly_comparison():
    """Monthly comparison analysis"""
    st.subheader("📅 Monthly Comparison")
    
    df = get_combined_data()
    
    if df.empty:
        return
    
    # Monthly summary
    df['month_name'] = df['date'].dt.strftime('%B %Y')
    monthly_summary = df.groupby(['month_name', 'type'])['amount'].sum().unstack(fill_value=0)
    
    if 'expense' in monthly_summary.columns and 'income' in monthly_summary.columns:
        monthly_summary['balance'] = monthly_summary['income'] - monthly_summary['expense']
        monthly_summary['savings_rate'] = (monthly_summary['balance'] / monthly_summary['income'] * 100).round(1)
    
    st.dataframe(monthly_summary, use_container_width=True)
    
    # Monthly trends
    if len(monthly_summary) > 1:
        fig = go.Figure()
        
        if 'income' in monthly_summary.columns:
            fig.add_trace(go.Scatter(
                x=monthly_summary.index,
                y=monthly_summary['income'],
                mode='lines+markers',
                name='Income',
                line=dict(color='green', width=3)
            ))
        
        if 'expense' in monthly_summary.columns:
            fig.add_trace(go.Scatter(
                x=monthly_summary.index,
                y=monthly_summary['expense'],
                mode='lines+markers',
                name='Expenses',
                line=dict(color='red', width=3)
            ))
        
        fig.update_layout(
            title='📈 Monthly Income vs Expenses Trend',
            xaxis_title='Month',
            yaxis_title='Amount (₹)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function for Financial Analysis & Insights page"""
    
    st.title("📊 Financial Analysis & Insights")
    st.markdown("---")
    
    # Check if data exists
    df = get_combined_data()
    if df.empty:
        st.error("❌ No financial data found! Please add data using the 'Data Input & Management' page first.")
        st.info("💡 Go to the Data Input page to add your financial data before viewing insights.")
        return
    
    # Overview Dashboard
    financial_overview_dashboard()
    st.markdown("---")
    
    # Smart Alerts
    spending_analysis_alerts()
    st.markdown("---")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📈 Category Analysis", "📅 Monthly View", "💡 Insights"])
    
    with tab1:
        st.subheader("📊 Financial Visualizations")
        
        # Income vs Expense chart
        income_expense_fig = create_expense_income_chart()
        if income_expense_fig:
            st.plotly_chart(income_expense_fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category pie chart
            pie_fig = create_category_pie_chart()
            if pie_fig:
                st.plotly_chart(pie_fig, use_container_width=True)
        
        with col2:
            # Summary stats
            st.subheader("📋 Quick Stats")
            total_transactions = len(df)
            date_range = (df['date'].max() - df['date'].min()).days
            avg_daily_expense = df[df['type'] == 'expense']['amount'].sum() / max(date_range, 1)
            
            st.metric("Total Transactions", total_transactions)
            st.metric("Days of Data", date_range)
            st.metric("Avg Daily Expense", f"₹{avg_daily_expense:.2f}")
        
        # Spending trend
        trend_fig = create_spending_trend_chart()
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True)
    
    with tab2:
        category_analysis()
    
    with tab3:
        monthly_comparison()
    
    with tab4:
        st.subheader("💡 Financial Insights & Recommendations")
        
        # Generate insights based on data
        total_income = df[df['type'] == 'income']['amount'].sum()
        total_expenses = df[df['type'] == 'expense']['amount'].sum()
        savings_rate = ((total_income - total_expenses) / total_income * 100) if total_income > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**💰 Savings Analysis:**")
            if savings_rate > 20:
                st.success(f"✅ Excellent! You're saving {savings_rate:.1f}% of your income.")
            elif savings_rate > 10:
                st.info(f"👍 Good savings rate of {savings_rate:.1f}%. Consider increasing to 20%+")
            elif savings_rate > 0:
                st.warning(f"⚠️ Low savings rate of {savings_rate:.1f}%. Try to reduce expenses.")
            else:
                st.error("❌ You're spending more than you earn! Review your expenses.")
        
        with col2:
            st.write("**📊 Expense Analysis:**")
            if not df[df['type'] == 'expense'].empty:
                top_category = df[df['type'] == 'expense'].groupby('category')['amount'].sum().idxmax()
                top_amount = df[df['type'] == 'expense'].groupby('category')['amount'].sum().max()
                percentage = (top_amount / total_expenses) * 100
                
                st.write(f"• **Highest expense category**: {top_category}")
                st.write(f"• **Amount**: ₹{top_amount:,.2f} ({percentage:.1f}%)")
                
                if percentage > 50:
                    st.warning(f"⚠️ {top_category} dominates your expenses. Consider budgeting.")

if __name__ == "__main__":
    main()
