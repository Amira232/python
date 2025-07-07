import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Electricity Consumption Calculator",
    page_icon="⚡",
    layout="wide"
)

# Title and header
st.title("⚡ Electricity Consumption Calculator")
st.markdown("Calculate your daily electricity consumption and track weekly equipment usage")

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 Basic Information")
    
    # User information inputs
    name = st.text_input("Enter your name:", placeholder="Your name")
    age = st.number_input("Enter your age:", min_value=1, max_value=120, value=25)
    city = st.text_input("Enter your city:", placeholder="City name")
    area = st.text_input("Enter your area name:", placeholder="Area/locality")
    
    # Housing type
    flat_tenament = st.selectbox("Are you living in:", ["Flat", "Tenement"])
    facility = st.selectbox("Select your accommodation type:", ["1BHK", "2BHK", "3BHK"])
    


with col2:
    st.header("📅 Weekly Equipment Usage Tracker")
    
    # Days of the week
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Create a dictionary to store weekly usage
    weekly_usage = {}
    
    # Create checkboxes for each day and equipment
    for day in days:
        st.subheader(f"{day}")
        col_ac, col_wm, col_fridge = st.columns(3)
        
        with col_ac:
            ac_check = st.checkbox(f"AC", key=f"ac_{day}")
        with col_wm:
            wm_check = st.checkbox(f"Washing Machine", key=f"wm_{day}")
        with col_fridge:
            fridge_check = st.checkbox(f"Fridge", key=f"fridge_{day}")
        
        weekly_usage[day] = {
            'AC': ac_check,
            'Washing Machine': wm_check,
            'Fridge': fridge_check
        }

# Calculate energy consumption
def calculate_energy():
    cal_energy = 0
    
    # Base consumption based on facility type
    if facility == "1BHK":
        cal_energy += 2 * 0.4 + 2 * 0.8  # 2.4 kWh
    elif facility == "2BHK":
        cal_energy += 3 * 0.4 + 3 * 0.8  # 3.6 kWh
    elif facility == "3BHK":
        cal_energy += 4 * 0.4 + 4 * 0.8  # 4.8 kWh
    
    # Add equipment consumption - no daily equipment usage
    
    return cal_energy

# Calculate weekly equipment usage
def calculate_weekly_usage():
    equipment_count = {'AC': 0, 'Washing Machine': 0, 'Fridge': 0}
    
    for day, usage in weekly_usage.items():
        for equipment, used in usage.items():
            if used:
                equipment_count[equipment] += 1
    
    return equipment_count

# Display results
if st.button("Calculate Consumption", type="primary"):
    if name and city and area:
        daily_consumption = calculate_energy()
        weekly_equipment_usage = calculate_weekly_usage()
        
        # Create three columns for results
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric("Daily Consumption", f"{daily_consumption:.1f} kWh")
        
        with result_col2:
            monthly_consumption = daily_consumption * 30
            st.metric("Monthly Consumption", f"{monthly_consumption:.1f} kWh")
        
        with result_col3:
            # Assuming average electricity cost of ₹5 per kWh
            monthly_cost = monthly_consumption * 5
            st.metric("Estimated Monthly Cost", f"₹{monthly_cost:.0f}")
        
        # Display user information
        st.header("👤 User Summary")
        st.info(f"""
        **Name:** {name}  
        **Age:** {age}  
        **Location:** {area}, {city}  
        **Housing:** {facility} {flat_tenament}  
        **Weekly Equipment Usage:** See analysis below
        """)
        
        # Weekly usage visualization
        st.header("📊 Weekly Equipment Usage Analysis")
        
        # Create bar chart for weekly equipment usage
        fig_bar = px.bar(
            x=list(weekly_equipment_usage.keys()),
            y=list(weekly_equipment_usage.values()),
            title="Equipment Usage Days per Week",
            labels={'x': 'Equipment', 'y': 'Days Used'},
            color=list(weekly_equipment_usage.values()),
            color_continuous_scale='viridis'
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Create heatmap for daily usage
        st.subheader("📈 Daily Usage Heatmap")
        
        # Prepare data for heatmap
        heatmap_data = []
        for day in days:
            row = []
            for equipment in ['AC', 'Washing Machine', 'Fridge']:
                row.append(1 if weekly_usage[day][equipment] else 0)
            heatmap_data.append(row)
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=['AC', 'Washing Machine', 'Fridge'],
            y=days,
            colorscale='RdYlGn',
            showscale=True
        ))
        fig_heatmap.update_layout(
            title="Weekly Equipment Usage Pattern",
            xaxis_title="Equipment",
            yaxis_title="Days"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Consumption breakdown
        st.header("💡 Consumption Breakdown")
        
        # Calculate component-wise consumption
        base_consumption = 0
        if facility == "1BHK":
            base_consumption = 2.4
        elif facility == "2BHK":
            base_consumption = 3.6
        elif facility == "3BHK":
            base_consumption = 4.8
        
        breakdown = {
            'Base Consumption': base_consumption
        }
        
        # Filter out zero values
        breakdown = {k: v for k, v in breakdown.items() if v > 0}
        
        fig_pie = px.pie(
            values=list(breakdown.values()),
            names=list(breakdown.keys()),
            title="Daily Energy Consumption Breakdown"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Weekly consumption summary
        st.header("📋 Weekly Summary")
        
        # Create a summary table
        weekly_summary = []
        for day in days:
            daily_extra = 0
            equipment_used = []
            
            for equipment in ['AC', 'Washing Machine', 'Fridge']:
                if weekly_usage[day][equipment]:
                    daily_extra += 3  # Each equipment adds 3 kWh
                    equipment_used.append(equipment)
            
            total_day_consumption = base_consumption + daily_extra
            weekly_summary.append({
                'Day': day,
                'Equipment Used': ', '.join(equipment_used) if equipment_used else 'Base only',
                'Daily Consumption (kWh)': total_day_consumption,
                'Daily Cost (₹)': total_day_consumption * 5
            })
        
        df_summary = pd.DataFrame(weekly_summary)
        st.dataframe(df_summary, use_container_width=True)
        
        # Total weekly consumption
        total_weekly = sum([row['Daily Consumption (kWh)'] for row in weekly_summary])
        total_weekly_cost = sum([row['Daily Cost (₹)'] for row in weekly_summary])
        
        st.success(f"""
        **Weekly Total:** {total_weekly:.1f} kWh  
        **Weekly Cost:** ₹{total_weekly_cost:.0f}  
        **Average Daily:** {total_weekly/7:.1f} kWh
        """)
        
    else:
        st.error("Please fill in all required fields (Name, City, Area)")

# Footer
st.markdown("---")
st.markdown("💡 **Note:** Energy consumption values are approximate. Actual consumption may vary based on usage patterns, appliance efficiency, and local conditions.")
st.markdown("💰 **Cost Calculation:** Based on average electricity rate of ₹5 per kWh. Actual rates may vary by location and utility provider.")
