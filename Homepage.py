import streamlit as st
import random
from datetime import datetime

def initialize_homepage_session():
    """Initialize homepage session state"""
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '',
            'setup_complete': False
        }

def create_welcome_section():
    """Create welcome section with enhanced design"""
    st.markdown("""
    <div style='
        text-align: center; 
        padding: 40px 30px; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); 
        border-radius: 20px; 
        margin-bottom: 40px; 
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;'>
        <div style='position: absolute; top: -50px; right: -50px; width: 100px; height: 100px; background: rgba(255,255,255,0.1); border-radius: 50%; opacity: 0.6;'></div>
        <div style='position: absolute; bottom: -30px; left: -30px; width: 80px; height: 80px; background: rgba(255,255,255,0.1); border-radius: 50%; opacity: 0.4;'></div>
        <div style='position: relative; z-index: 1;'>
            <h1 style='margin: 0; font-size: 3.2em; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); animation: fadeInUp 1s ease-out;'>
                🤖 AI-Powered Financial Advisor
            </h1>
            <h2 style='margin: 20px 0; font-weight: 300; font-size: 1.8em; opacity: 0.95; animation: fadeInUp 1s ease-out 0.2s both;'>
                Your Personal Finance Management Solution
            </h2>
            <p style='margin: 0; font-size: 1.2em; opacity: 0.9; max-width: 600px; margin: 0 auto; line-height: 1.6; animation: fadeInUp 1s ease-out 0.4s both;'>
                Manage your finances smartly with AI-driven insights, real-time analysis, and personalized recommendations
            </p>
            <div style='margin-top: 30px; animation: fadeInUp 1s ease-out 0.6s both;'>
                <span style='display: inline-block; padding: 8px 20px; background: rgba(255,255,255,0.2); border-radius: 20px; font-size: 0.9em; margin: 0 5px;'>✨ AI-Powered</span>
                <span style='display: inline-block; padding: 8px 20px; background: rgba(255,255,255,0.2); border-radius: 20px; font-size: 0.9em; margin: 0 5px;'>🔒 Secure</span>
                <span style='display: inline-block; padding: 8px 20px; background: rgba(255,255,255,0.2); border-radius: 20px; font-size: 0.9em; margin: 0 5px;'>🚀 Easy to Use</span>
            </div>
        </div>
    </div>
    <style>
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
    """, unsafe_allow_html=True)

def create_platform_stats():
    """Create platform statistics with modern cards"""
    st.markdown("""
    <div style='margin: 40px 0;'>
        <h2 style='text-align: center; color: #2c3e50; margin-bottom: 30px; font-weight: 600;'>📊 Platform Highlights</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    stats = [
        {"icon": "📝", "title": "Features", "value": "5", "subtitle": "Complete Suite", "color": "#e74c3c"},
        {"icon": "🤖", "title": "AI Powered", "value": "100%", "subtitle": "Smart Insights", "color": "#3498db"},
        {"icon": "🔐", "title": "Security", "value": "Bank Level", "subtitle": "Your Data Safe", "color": "#2ecc71"},
        {"icon": "📱", "title": "Access", "value": "24/7", "subtitle": "All Devices", "color": "#f39c12"}
    ]
    
    cols = [col1, col2, col3, col4]
    
    for i, stat in enumerate(stats):
        with cols[i]:
            st.markdown(f"""
            <div style='
                text-align: center;
                padding: 30px 20px;
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border-radius: 20px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                border: 1px solid rgba(0,0,0,0.05);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                cursor: pointer;
                position: relative;
                overflow: hidden;
            ' onmouseover='this.style.transform="translateY(-5px)"; this.style.boxShadow="0 15px 35px rgba(0,0,0,0.15)";' 
               onmouseout='this.style.transform="translateY(0)"; this.style.boxShadow="0 8px 25px rgba(0,0,0,0.1)";'>
                <div style='position: absolute; top: -20px; right: -20px; width: 40px; height: 40px; background: {stat["color"]}; opacity: 0.1; border-radius: 50%;'></div>
                <div style='font-size: 3em; margin-bottom: 10px; color: {stat["color"]};'>{stat["icon"]}</div>
                <h3 style='margin: 0; color: #2c3e50; font-weight: 600; font-size: 1.1em;'>{stat["title"]}</h3>
                <div style='font-size: 2em; font-weight: 700; color: {stat["color"]}; margin: 10px 0;'>{stat["value"]}</div>
                <p style='margin: 0; color: #7f8c8d; font-size: 0.9em; font-weight: 500;'>{stat["subtitle"]}</p>
            </div>
            """, unsafe_allow_html=True)

def create_feature_overview():
    """Create enhanced feature overview cards"""
    st.markdown("""
    <div style='margin: 50px 0 30px 0;'>
        <h2 style='text-align: center; color: #2c3e50; margin-bottom: 15px; font-weight: 600; font-size: 2.5em;'>🚀 Powerful Features</h2>
        <p style='text-align: center; color: #7f8c8d; font-size: 1.2em; max-width: 600px; margin: 0 auto;'>
            Discover our comprehensive suite of AI-powered financial tools designed to transform your money management experience
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    features = [
        {
            'title': 'Data Management',
            'description': 'Effortlessly manage your financial data with our intuitive interface. Upload CSV files, enter data manually, and ensure accuracy with built-in validation.',
            'page': 'Data Input & Management',
            'icon': '📊',
            'color': '#e74c3c',
            'gradient': 'linear-gradient(135deg, #ff6b6b 0%, #e74c3c 100%)',
            'benefits': ['Manual Entry', 'CSV Upload', 'Data Validation', 'Export Options']
        },
        {
            'title': 'Financial Analysis',
            'description': 'Get deep insights into your spending patterns with AI-powered analysis. Receive smart alerts and visualize your financial health through interactive charts.',
            'page': 'Financial Analysis & Insights',
            'icon': '📈',
            'color': '#00d2d3',
            'gradient': 'linear-gradient(135deg, #00d2d3 0%, #00a8a9 100%)',
            'benefits': ['Smart Alerts', 'Visual Charts', 'Spending Analysis', 'Monthly Reports']
        },
        {
            'title': 'Goal Management',
            'description': 'Set ambitious financial goals and track your progress with precision. Our AI provides personalized recommendations to help you achieve your dreams faster.',
            'page': 'Goal Management',
            'icon': '🎯',
            'color': '#3498db',
            'gradient': 'linear-gradient(135deg, #74b9ff 0%, #3498db 100%)',
            'benefits': ['Goal Setting', 'Progress Tracking', 'Calendar View', 'AI Recommendations']
        },
        {
            'title': 'AI Chat Advisor',
            'description': 'Experience the future of financial advice with our intelligent chatbot. Get instant answers, voice support, and multilingual assistance whenever you need it.',
            'page': 'AI Financial Advisor Chat',
            'icon': '🤖',
            'color': '#00b894',
            'gradient': 'linear-gradient(135deg, #00b894 0%, #00a085 100%)',
            'benefits': ['AI Conversations', 'Voice Input', 'Multi-language', 'Real-time Advice']
        },
        {
            'title': 'Investment Insights',
            'description': 'Make informed investment decisions with real-time market data, predictive analytics, and personalized portfolio recommendations from our AI engine.',
            'page': 'Investment Insights',
            'icon': '💎',
            'color': '#fdcb6e',
            'gradient': 'linear-gradient(135deg, #fdcb6e 0%, #e17055 100%)',
            'benefits': ['Market Data', 'Stock Analysis', 'Portfolio Tracking', 'Predictions']
        }
    ]
    
    for i, feature in enumerate(features):
        # Alternate layout - left/right
        if i % 2 == 0:
            col1, col2 = st.columns([1, 1])
            content_col, image_col = col1, col2
        else:
            col1, col2 = st.columns([1, 1])
            image_col, content_col = col1, col2
        
        with image_col:
            st.markdown(f"""
            <div style='
                text-align: center;
                padding: 40px;
                background: {feature["gradient"]};
                border-radius: 20px;
                margin: 20px 0;
                color: white;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            '>
                <div style='font-size: 5em; margin-bottom: 20px;'>{feature["icon"]}</div>
                <h3 style='margin: 0; font-size: 1.8em; font-weight: 600;'>{feature["title"]}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with content_col:
            st.markdown(f"""
            <div style='
                padding: 30px;
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border-radius: 20px;
                margin: 20px 0;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                border-left: 5px solid {feature["color"]};
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
            '>
                <h3 style='margin: 0 0 15px 0; color: {feature["color"]}; font-size: 1.8em; font-weight: 700;'>
                    {feature["title"]}
                </h3>
                <p style='margin: 0 0 20px 0; color: #495057; line-height: 1.7; font-size: 1.1em;'>
                    {feature["description"]}
                </p>
                <div style='display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;'>
                    {"".join([f"<span style='background: linear-gradient(135deg, {feature['color']}20, {feature['color']}10); color: {feature['color']}; padding: 6px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; border: 1px solid {feature['color']}30;'>✓ {benefit}</span>" for benefit in feature['benefits']])}
                </div>
                <div style='
                    padding: 15px;
                    background: linear-gradient(135deg, {feature["color"]}10, {feature["color"]}05);
                    border-radius: 10px;
                    border-left: 3px solid {feature["color"]};
                '>
                    <small style='color: #6c757d; font-weight: 600;'>
                        🚀 Navigate to: <strong style='color: {feature["color"]};'>{feature["page"]}</strong>
                    </small>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if i < len(features) - 1:
            st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)

def create_quick_actions():
    """Create enhanced quick action buttons"""
    st.markdown("""
    <div style='margin: 40px 0 20px 0;'>
        <h2 style='color: #2c3e50; margin-bottom: 15px; font-weight: 600; font-size: 2.2em;'>⚡ Quick Start</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        color: white;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    '>
        <h3 style='margin: 0 0 10px 0; font-weight: 600;'>🚀 Ready to Transform Your Financial Future?</h3>
        <p style='margin: 0; opacity: 0.95; line-height: 1.6;'>
            Take the first step towards financial freedom. Choose an action below to begin your personalized financial journey with AI-powered insights!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced buttons with better styling
    col1, col2 = st.columns(2, gap="medium")
    
    actions = [
        {"text": "📝 Start Adding Data", "color": "#e74c3c", "desc": "Begin by adding your financial transactions"},
        {"text": "🎯 Set Financial Goals", "color": "#3498db", "desc": "Define and track your financial objectives"},
        {"text": "💬 Chat with AI Advisor", "color": "#00b894", "desc": "Get instant personalized financial advice"},
        {"text": "📈 Explore Investments", "color": "#fdcb6e", "desc": "Discover investment opportunities"}
    ]
    
    cols = [col1, col2, col1, col2]
    
    for i, action in enumerate(actions):
        with cols[i]:
            button_key = f"action_button_{i}"
            
            # Create custom styled button using markdown
            st.markdown(f"""
            <div style='margin-bottom: 15px;'>
                <div style='
                    background: linear-gradient(135deg, {action["color"]} 0%, {action["color"]}dd 100%);
                    padding: 20px;
                    border-radius: 15px;
                    text-align: center;
                    color: white;
                    box-shadow: 0 8px 25px {action["color"]}40;
                    transition: all 0.3s ease;
                    cursor: pointer;
                    border: none;
                ' onmouseover='this.style.transform="translateY(-3px)"; this.style.boxShadow="0 12px 30px {action["color"]}50";'
                   onmouseout='this.style.transform="translateY(0)"; this.style.boxShadow="0 8px 25px {action["color"]}40";'>
                    <h4 style='margin: 0 0 8px 0; font-weight: 600; font-size: 1.2em;'>{action["text"]}</h4>
                    <p style='margin: 0; opacity: 0.9; font-size: 0.9em;'>{action["desc"]}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Invisible button for functionality
            if st.button(f"_{action['text']}", key=button_key, help=action["desc"]):
                if i == 0:
                    # st.balloons()
                    st.success("🎉 Excellent choice! Navigate to 'Data Input & Management' page using the sidebar to add your financial data.")
                elif i == 1:
                    st.success("💡 Navigate to 'Goal Management' page to create and track your financial goals.")
                elif i == 2:
                    st.success("🤖 Navigate to 'AI Financial Advisor Chat' for personalized financial advice.")
                else:
                    st.success("💎 Navigate to 'Investment Insights' to discover investment opportunities.")

def create_how_it_works():
    """Create enhanced how it works section"""
    st.markdown("""
    <div style='margin: 60px 0 40px 0; text-align: center;'>
        <h2 style='color: #2c3e50; margin-bottom: 15px; font-weight: 600; font-size: 2.5em;'>🔄 How It Works</h2>
        <p style='color: #7f8c8d; font-size: 1.2em; max-width: 600px; margin: 0 auto; line-height: 1.6;'>
            Our AI-powered platform guides you through a simple 4-step process to achieve your financial goals
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    steps = [
        {
            'step': 1,
            'title': 'Add Your Data',
            'description': 'Securely upload your financial data via CSV or enter transactions manually with our intuitive interface',
            'icon': '📊',
            'color': '#e74c3c'
        },
        {
            'step': 2,
            'title': 'AI Analysis',
            'description': 'Our advanced AI analyzes your spending patterns, identifies trends, and assesses your financial health',
            'icon': '🤖',
            'color': '#3498db'
        },
        {
            'step': 3,
            'title': 'Set Smart Goals',
            'description': 'Define personalized financial goals and let our AI create actionable roadmaps for achievement',
            'icon': '🎯',
            'color': '#00b894'
        },
        {
            'step': 4,
            'title': 'Get Insights',
            'description': 'Receive personalized recommendations, investment advice, and real-time guidance to optimize your finances',
            'icon': '💡',
            'color': '#fdcb6e'
        }
    ]
    
    cols = st.columns(4)
    
    for i, step in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div style='
                text-align: center;
                padding: 30px 20px;
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                margin: 20px 0;
                position: relative;
                transition: transform 0.3s ease;
                border-top: 4px solid {step["color"]};
                height: 350px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            ' onmouseover='this.style.transform="translateY(-8px)";' 
               onmouseout='this.style.transform="translateY(0)";'>
                <div>
                    <div style='
                        display: inline-block;
                        background: {step["color"]};
                        color: white;
                        width: 40px;
                        height: 40px;
                        border-radius: 50%;
                        line-height: 40px;
                        font-weight: bold;
                        font-size: 1.2em;
                        margin-bottom: 15px;
                    '>{step["step"]}</div>
                    <div style='font-size: 3.5em; margin-bottom: 15px; color: {step["color"]};'>{step["icon"]}</div>
                    <h4 style='margin: 0 0 15px 0; color: #2c3e50; font-weight: 700; font-size: 1.3em;'>{step["title"]}</h4>
                </div>
                <p style='margin: 0; color: #7f8c8d; font-size: 0.95em; line-height: 1.5; text-align: center;'>
                    {step["description"]}
                </p>
            </div>
            """, unsafe_allow_html=True)

def create_financial_tips():
    """Create enhanced financial tips section"""
    st.markdown("""
    <div style='margin: 40px 0 20px 0;'>
        <h2 style='color: #2c3e50; margin-bottom: 15px; font-weight: 600; font-size: 2.2em;'>💡 Expert Financial Tips</h2>
    </div>
    """, unsafe_allow_html=True)
    
    tips = [
        {
            'tip': "💰 Emergency Fund Strategy",
            'description': "Build an emergency fund covering 6-12 months of expenses. Start with small amounts and automate your savings to reach this crucial financial safety net.",
            'color': '#e74c3c'
        },
        {
            'tip': "📊 The 50-30-20 Formula", 
            'description': "Allocate 50% of income for needs, 30% for wants, and 20% for savings and investments. This balanced approach ensures both enjoyment and financial growth.",
            'color': '#3498db'
        },
        {
            'tip': "📈 Compound Interest Power",
            'description': "Start investing early to harness compound interest. Even small amounts invested consistently can grow into substantial wealth over time.",
            'color': '#00b894'
        },
        {
            'tip': "🔄 Diversification Principle",
            'description': "Spread your investments across different asset classes, sectors, and geographies to reduce risk and optimize returns in your portfolio.",
            'color': '#9b59b6'
        },
        {
            'tip': "📚 Continuous Learning",
            'description': "Stay informed about financial markets, economic trends, and investment strategies. Knowledge is your best tool for making smart money decisions.",
            'color': '#f39c12'
        },
        {
            'tip': "🎯 SMART Goal Setting",
            'description': "Create Specific, Measurable, Achievable, Relevant, and Time-bound financial goals. Clear objectives lead to better financial outcomes.",
            'color': '#1abc9c'
        },
        {
            'tip': "💳 Expense Tracking Habit",
            'description': "Monitor every rupee spent to identify patterns and opportunities for optimization. Small savings can lead to significant wealth accumulation.",
            'color': '#e67e22'
        },
        {
            'tip': "🏦 Automation Benefits",
            'description': "Automate your savings, investments, and bill payments. This ensures consistency and removes the temptation to spend money earmarked for savings.",
            'color': '#8e44ad'
        }
    ]
    
    # Select random tips to display
    selected_tips = random.sample(tips, 3)
    
    for tip_data in selected_tips:
        st.markdown(f"""
        <div style='
            padding: 25px;
            margin: 20px 0;
            border-radius: 15px;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-left: 5px solid {tip_data["color"]};
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
            position: relative;
            overflow: hidden;
        ' onmouseover='this.style.transform="translateX(5px)";' 
           onmouseout='this.style.transform="translateX(0)";'>
            <div style='position: absolute; top: -20px; right: -20px; width: 80px; height: 80px; background: {tip_data["color"]}; opacity: 0.05; border-radius: 50%;'></div>
            <h4 style='margin: 0 0 12px 0; color: {tip_data["color"]}; font-weight: 700; font-size: 1.3em; display: flex; align-items: center;'>
                {tip_data["tip"]}
            </h4>
            <p style='margin: 0; color: #495057; line-height: 1.7; font-size: 1.05em;'>
                {tip_data["description"]}
            </p>
        </div>
        """, unsafe_allow_html=True)

def create_benefits_section():
    """Create enhanced benefits section"""
    st.markdown("""
    <div style='margin: 60px 0 40px 0; text-align: center;'>
        <h2 style='color: #2c3e50; margin-bottom: 15px; font-weight: 600; font-size: 2.5em;'>✨ Why Choose Our AI Financial Advisor?</h2>
        <p style='color: #7f8c8d; font-size: 1.2em; max-width: 700px; margin: 0 auto; line-height: 1.6;'>
            Experience the future of personal finance management with cutting-edge AI technology and user-centric design
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        benefits = [
            {
                "icon": "🤖",
                "title": "AI-Powered Insights",
                "desc": "Get intelligent analysis of your financial patterns with machine learning algorithms",
                "color": "#3498db"
            },
            {
                "icon": "📊", 
                "title": "Comprehensive Tracking",
                "desc": "Monitor income, expenses, and investments in one unified, intuitive dashboard",
                "color": "#e74c3c"
            },
            {
                "icon": "🎯",
                "title": "Goal Achievement", 
                "desc": "Set and track financial goals with AI-powered progress monitoring and optimization",
                "color": "#00b894"
            },
            {
                "icon": "🔒",
                "title": "Bank-Level Security",
                "desc": "Your financial data stays protected with enterprise-grade security measures",
                "color": "#9b59b6"
            }
        ]
        
        for benefit in benefits:
            st.markdown(f"""
            <div style='
                display: flex;
                align-items: flex-start;
                padding: 20px;
                margin: 15px 0;
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border-radius: 12px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                border-left: 4px solid {benefit["color"]};
                transition: transform 0.3s ease;
            ' onmouseover='this.style.transform="translateX(5px)";' 
               onmouseout='this.style.transform="translateX(0)";'>
                <div style='
                    font-size: 2.5em;
                    margin-right: 15px;
                    color: {benefit["color"]};
                    min-width: 60px;
                '>{benefit["icon"]}</div>
                <div>
                    <h4 style='margin: 0 0 8px 0; color: #2c3e50; font-weight: 600; font-size: 1.2em;'>{benefit["title"]}</h4>
                    <p style='margin: 0; color: #7f8c8d; line-height: 1.6;'>{benefit["desc"]}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        more_benefits = [
            {
                "icon": "💬",
                "title": "24/7 AI Support", 
                "desc": "Get financial advice anytime through our intelligent chatbot with natural language processing",
                "color": "#1abc9c"
            },
            {
                "icon": "📈",
                "title": "Investment Guidance",
                "desc": "Real-time market data and personalized investment recommendations powered by AI analysis",
                "color": "#f39c12"
            },
            {
                "icon": "🌐",
                "title": "Multi-Language Support",
                "desc": "Access financial advice in multiple Indian languages with seamless translation capabilities",
                "color": "#e67e22"
            },
            {
                "icon": "📱",
                "title": "Universal Access",
                "desc": "Responsive design that works perfectly on all devices - mobile, tablet, and desktop",
                "color": "#8e44ad"
            }
        ]
        
        for benefit in more_benefits:
            st.markdown(f"""
            <div style='
                display: flex;
                align-items: flex-start;
                padding: 20px;
                margin: 15px 0;
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border-radius: 12px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                border-left: 4px solid {benefit["color"]};
                transition: transform 0.3s ease;
            ' onmouseover='this.style.transform="translateX(5px)";' 
               onmouseout='this.style.transform="translateX(0)";'>
                <div style='
                    font-size: 2.5em;
                    margin-right: 15px;
                    color: {benefit["color"]};
                    min-width: 60px;
                '>{benefit["icon"]}</div>
                <div>
                    <h4 style='margin: 0 0 8px 0; color: #2c3e50; font-weight: 600; font-size: 1.2em;'>{benefit["title"]}</h4>
                    <p style='margin: 0; color: #7f8c8d; line-height: 1.6;'>{benefit["desc"]}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

def create_footer():
    """Create enhanced footer section"""
    st.markdown("""
    <div style='margin: 60px 0 40px 0;'>
        <div style='
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            padding: 40px 30px;
            border-radius: 20px;
            color: white;
            box-shadow: 0 10px 30px rgba(44, 62, 80, 0.3);
        '>
            <div style='text-align: center; margin-bottom: 40px;'>
                <h2 style='margin: 0 0 10px 0; font-weight: 600; font-size: 2.2em;'>🤖 AI-Powered Financial Advisor</h2>
                <p style='margin: 0; font-size: 1.2em; opacity: 0.9;'>Your journey to financial freedom starts here</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main function for Homepage with enhanced styling"""
    
    st.set_page_config(
        page_title="AI Financial Advisor - Transform Your Financial Future",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
        .main > div {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        .stButton > button {
            display: none !important;
        }
        
        .element-container {
            margin-bottom: 1rem;
        }
        
        @media (max-width: 768px) {
            .main > div {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_homepage_session()
    
    # Welcome Section
    create_welcome_section()
    
    # Platform Stats
    create_platform_stats()
    
    # Feature Overview
    create_feature_overview()
    
    # How it works
    create_how_it_works()
    
    # Two column layout for actions and tips
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        create_quick_actions()
    
    with col2:
        create_financial_tips()
    
    # Benefits Section
    create_benefits_section()
    
    # Footer
    create_footer()

if __name__ == "__main__":
    main()
