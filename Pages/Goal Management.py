import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import calendar
from dateutil.relativedelta import relativedelta

def initialize_goals_session_state():
    """Initialize session state for goals"""
    if 'financial_goals' not in st.session_state:
        st.session_state.financial_goals = []

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
        uploaded_df.columns = uploaded_df.columns.str.lower()
        combined_df = pd.concat([combined_df, uploaded_df], ignore_index=True)
    
    if not combined_df.empty:
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df['amount'] = pd.to_numeric(combined_df['amount'])
        combined_df['type'] = combined_df['type'].str.lower()
        combined_df = combined_df.sort_values('date')
    
    return combined_df

def calculate_savings_capacity():
    """Calculate user's savings capacity based on historical data"""
    df = get_combined_data()
    
    if df.empty:
        return 0, 0, 0
    
    # Calculate monthly averages
    df['month'] = df['date'].dt.to_period('M')
    monthly_summary = df.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
    
    if 'income' not in monthly_summary.columns or 'expense' not in monthly_summary.columns:
        return 0, 0, 0
    
    avg_monthly_income = monthly_summary['income'].mean()
    avg_monthly_expense = monthly_summary['expense'].mean()
    avg_monthly_savings = avg_monthly_income - avg_monthly_expense
    
    return avg_monthly_income, avg_monthly_expense, max(0, avg_monthly_savings)

def create_new_goal():
    """Create new financial goal form"""
    st.subheader("🎯 Create New Financial Goal")
    
    with st.form("new_goal_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            goal_name = st.text_input("Goal Name", placeholder="e.g., Emergency Fund, Vacation, New Car")
            target_amount = st.number_input("Target Amount (₹)", min_value=1000, step=1000)
            goal_category = st.selectbox("Goal Category", [
                "Emergency Fund", "Vacation", "Education", "Home Purchase", 
                "Vehicle", "Investment", "Wedding", "Healthcare", "Other"
            ])
        
        with col2:
            target_date = st.date_input("Target Date", min_value=date.today() + timedelta(days=30))
            current_amount = st.number_input("Current Saved Amount (₹)", min_value=0, step=500)
            priority = st.selectbox("Priority", ["High", "Medium", "Low"])
        
        description = st.text_area("Description (Optional)", placeholder="Add notes about this goal...")
        
        # Calculate suggested monthly savings
        months_to_goal = ((target_date.year - date.today().year) * 12 + 
                         (target_date.month - date.today().month))
        
        if months_to_goal > 0 and target_amount > current_amount:
            remaining_amount = target_amount - current_amount
            suggested_monthly = remaining_amount / months_to_goal
            st.info(f"💡 Suggested monthly savings: ₹{suggested_monthly:,.2f} over {months_to_goal} months")
        
        submitted = st.form_submit_button("🎯 Create Goal", type="primary")
        
        if submitted:
            if goal_name and target_amount > 0:
                new_goal = {
                    'id': len(st.session_state.financial_goals) + 1,
                    'name': goal_name,
                    'target_amount': target_amount,
                    'current_amount': current_amount,
                    'target_date': target_date.isoformat(),
                    'category': goal_category,
                    'priority': priority,
                    'description': description,
                    'created_date': date.today().isoformat(),
                    'status': 'Active',
                    'milestones': []
                }
                
                st.session_state.financial_goals.append(new_goal)
                st.success(f"✅ Goal '{goal_name}' created successfully!")
                st.rerun()
            else:
                st.error("Please fill in all required fields")

def display_active_goals():
    """Display and manage active goals"""
    st.subheader("📋 Your Active Goals")
    
    if not st.session_state.financial_goals:
        st.info("🎯 No goals created yet. Create your first financial goal above!")
        return
    
    active_goals = [goal for goal in st.session_state.financial_goals if goal['status'] == 'Active']
    
    if not active_goals:
        st.info("🎯 No active goals. All goals have been completed!")
        return
    
    for i, goal in enumerate(active_goals):
        with st.expander(f"🎯 {goal['name']} - {goal['priority']} Priority", expanded=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Progress calculation
                progress = (goal['current_amount'] / goal['target_amount']) * 100
                remaining = goal['target_amount'] - goal['current_amount']
                
                st.progress(progress / 100)
                st.write(f"**Progress:** {progress:.1f}% completed")
                st.write(f"**Target:** ₹{goal['target_amount']:,.2f}")
                st.write(f"**Saved:** ₹{goal['current_amount']:,.2f}")
                st.write(f"**Remaining:** ₹{remaining:,.2f}")
                
                # Time calculations
                target_date = datetime.fromisoformat(goal['target_date']).date()
                days_remaining = (target_date - date.today()).days
                
                if days_remaining > 0:
                    st.write(f"**Time Left:** {days_remaining} days")
                    if remaining > 0:
                        daily_savings_needed = remaining / days_remaining
                        st.write(f"**Daily savings needed:** ₹{daily_savings_needed:.2f}")
                else:
                    st.error("⏰ Goal deadline has passed!")
            
            with col2:
                st.write(f"**Category:** {goal['category']}")
                st.write(f"**Target Date:** {goal['target_date']}")
                st.write(f"**Created:** {goal['created_date']}")
                
                if goal['description']:
                    st.write(f"**Notes:** {goal['description']}")
            
            with col3:
                # Goal actions
                st.write("**Actions:**")
                
                # Update savings
                new_amount = st.number_input(
                    f"Update saved amount", 
                    min_value=0, 
                    value=goal['current_amount'],
                    key=f"update_amount_{goal['id']}"
                )
                
                if st.button(f"💰 Update", key=f"update_{goal['id']}"):
                    st.session_state.financial_goals[i]['current_amount'] = new_amount
                    if new_amount >= goal['target_amount']:
                        st.session_state.financial_goals[i]['status'] = 'Completed'
                        st.success(f"🎉 Goal '{goal['name']}' completed!")
                    else:
                        st.success("💰 Amount updated!")
                    st.rerun()
                
                if st.button(f"🗑️ Delete", key=f"delete_{goal['id']}", type="secondary"):
                    st.session_state.financial_goals = [g for g in st.session_state.financial_goals if g['id'] != goal['id']]
                    st.success("🗑️ Goal deleted!")
                    st.rerun()

def goal_recommendations():
    """Provide AI-like goal recommendations"""
    st.subheader("💡 Smart Goal Recommendations")
    
    df = get_combined_data()
    avg_income, avg_expense, avg_savings = calculate_savings_capacity()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**💰 Based on Your Financial Data:**")
        
        if avg_savings > 0:
            st.success(f"✅ Average monthly savings: ₹{avg_savings:,.2f}")
            
            # Emergency fund recommendation
            emergency_fund = avg_expense * 6
            st.write(f"🏥 **Emergency Fund Goal:** ₹{emergency_fund:,.2f}")
            st.write(f"   (6 months of expenses)")
            
            # Vacation fund
            vacation_fund = avg_income * 0.1 * 12  # 10% of annual income
            st.write(f"✈️ **Annual Vacation Fund:** ₹{vacation_fund:,.2f}")
            
            # Investment goal
            investment_goal = avg_savings * 0.7 * 12  # 70% of savings for investment
            st.write(f"📈 **Annual Investment Goal:** ₹{investment_goal:,.2f}")
            
        else:
            st.warning("⚠️ Consider reducing expenses to create savings capacity for goals!")
            st.write("💡 Try to save at least 20% of your income")
    
    with col2:
        st.write("**🎯 Goal Setting Tips:**")
        
        tips = [
            "🎯 Set SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound)",
            "📊 Start with small, achievable goals to build momentum",
            "🏥 Prioritize emergency fund (3-6 months expenses)",
            "💰 Use the 50/30/20 rule: 50% needs, 30% wants, 20% savings",
            "📈 Consider inflation when setting long-term goals",
            "🔄 Review and adjust goals quarterly"
        ]
        
        for tip in tips:
            st.write(f"• {tip}")

def create_goal_progress_chart():
    """Create goal progress visualization"""
    if not st.session_state.financial_goals:
        return None
    
    goals_data = []
    for goal in st.session_state.financial_goals:
        progress = (goal['current_amount'] / goal['target_amount']) * 100
        goals_data.append({
            'Goal': goal['name'],
            'Progress': min(progress, 100),
            'Target': goal['target_amount'],
            'Current': goal['current_amount'],
            'Status': goal['status']
        })
    
    df = pd.DataFrame(goals_data)
    
    fig = px.bar(
        df,
        x='Progress',
        y='Goal',
        orientation='h',
        title='🎯 Goal Progress Overview',
        color='Progress',
        color_continuous_scale=['red', 'yellow', 'green'],
        text='Progress'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    fig.update_layout(
        xaxis_title="Progress (%)",
        yaxis_title="Goals",
        coloraxis_colorbar=dict(title="Progress %"),
        height=max(300, len(df) * 50)
    )
    
    return fig

def calendar_view():
    """Calendar view for goal deadlines and milestones"""
    st.subheader("📅 Goal Calendar View")
    
    if not st.session_state.financial_goals:
        st.info("📅 No goals to display in calendar view.")
        return
    
    # Create calendar data
    calendar_events = []
    for goal in st.session_state.financial_goals:
        target_date = datetime.fromisoformat(goal['target_date']).date()
        calendar_events.append({
            'date': target_date,
            'event': f"🎯 {goal['name']} Target",
            'amount': f"₹{goal['target_amount']:,.2f}",
            'priority': goal['priority'],
            'status': goal['status']
        })
    
    # Sort events by date
    calendar_events.sort(key=lambda x: x['date'])
    
    # Display upcoming deadlines
    st.write("**🔔 Upcoming Goal Deadlines:**")
    
    today = date.today()
    upcoming_events = [event for event in calendar_events if event['date'] >= today][:5]
    
    if upcoming_events:
        for event in upcoming_events:
            days_left = (event['date'] - today).days
            
            if days_left == 0:
                st.error(f"🚨 TODAY: {event['event']} - {event['amount']}")
            elif days_left <= 7:
                st.warning(f"⚠️ {days_left} days: {event['event']} - {event['amount']}")
            elif days_left <= 30:
                st.info(f"📅 {days_left} days: {event['event']} - {event['amount']}")
            else:
                st.write(f"📅 {days_left} days: {event['event']} - {event['amount']}")
    else:
        st.success("✅ No upcoming goal deadlines!")
    
    # Monthly calendar view
    selected_month = st.selectbox(
        "Select Month to View",
        options=pd.date_range(start=today, periods=12, freq='ME').strftime('%B %Y').tolist(),
        index=0
    )
    
    # Filter events for selected month
    selected_date = datetime.strptime(selected_month, '%B %Y').date()
    month_events = [
        event for event in calendar_events 
        if event['date'].year == selected_date.year and event['date'].month == selected_date.month
    ]
    
    if month_events:
        st.write(f"**📅 Goals in {selected_month}:**")
        for event in month_events:
            priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            status_icon = {"Active": "🎯", "Completed": "✅", "Paused": "⏸️"}
            
            st.write(f"• **{event['date']}** {priority_color[event['priority']]} {status_icon[event['status']]} {event['event']} - {event['amount']}")
    else:
        st.info(f"📅 No goals scheduled for {selected_month}")

def goal_analytics():
    """Analytics and insights for goals"""
    st.subheader("📊 Goal Analytics")
    
    if not st.session_state.financial_goals:
        st.info("📊 No goals data available for analytics.")
        return
    
    # Goal statistics
    total_goals = len(st.session_state.financial_goals)
    active_goals = len([g for g in st.session_state.financial_goals if g['status'] == 'Active'])
    completed_goals = len([g for g in st.session_state.financial_goals if g['status'] == 'Completed'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Goals", total_goals)
    with col2:
        st.metric("Active Goals", active_goals)
    with col3:
        st.metric("Completed Goals", completed_goals)
    with col4:
        completion_rate = (completed_goals / total_goals * 100) if total_goals > 0 else 0
        st.metric("Completion Rate", f"{completion_rate:.1f}%")
    
    # Goals by category
    if st.session_state.financial_goals:
        goals_df = pd.DataFrame(st.session_state.financial_goals)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            category_counts = goals_df['category'].value_counts()
            fig_cat = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="🥧 Goals by Category"
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            # Priority distribution
            priority_counts = goals_df['priority'].value_counts()
            fig_priority = px.bar(
                x=priority_counts.index,
                y=priority_counts.values,
                title="📊 Goals by Priority",
                color=priority_counts.index,
                color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            )
            fig_priority.update_layout(showlegend=False)
            st.plotly_chart(fig_priority, use_container_width=True)
        
        # Total amounts
        total_target = goals_df['target_amount'].sum()
        total_saved = goals_df['current_amount'].sum()
        total_remaining = total_target - total_saved
        
        st.write("**💰 Financial Overview:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Target", f"₹{total_target:,.2f}")
        with col2:
            st.metric("Total Saved", f"₹{total_saved:,.2f}")
        with col3:
            st.metric("Remaining", f"₹{total_remaining:,.2f}")

def main():
    """Main function for Goal Management page"""
    
    st.title("🎯 Goal Management & Tracking")
    st.markdown("---")
    
    # Initialize session state
    initialize_goals_session_state()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Goals", "💡 Recommendations", "📅 Calendar", "📊 Analytics"])
    
    with tab1:
        # Create new goal
        create_new_goal()
        st.markdown("---")
        
        # Display active goals
        display_active_goals()
        
        # Goal progress chart
        if st.session_state.financial_goals:
            st.markdown("---")
            progress_chart = create_goal_progress_chart()
            if progress_chart:
                st.plotly_chart(progress_chart, use_container_width=True)
    
    with tab2:
        goal_recommendations()
    
    with tab3:
        calendar_view()
    
    with tab4:
        goal_analytics()

if __name__ == "__main__":
    main()
