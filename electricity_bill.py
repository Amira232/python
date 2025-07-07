import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Electricity Consumption Calculator",
    page_icon="⚡",
    layout="wide"
)

# Title and header
st.title("⚡ Electricity Consumption Calculator")
st.markdown("Calculate your daily electricity consumption and track weekly equipment usage")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 Basic Information")
    name = st.text_input("Enter your name:", placeholder="Your name")
    age = st.number_input("Enter your age:", min_value=1, max_value=120, value=25)
    city = st.text_input("Enter your city:", placeholder="City name")
    area = st.text_input("Enter your area name:", placeholder="Area/locality")
    flat_tenament = st.selectbox("Are you living in:", ["Flat", "Tenement"])
    facility = st.selectbox("Select your accommodation type:", ["1BHK", "2BHK", "3BHK"])

with col2:
    st.header("📅 Weekly Equipment Usage Tracker")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly_usage = {}

    for day in days:
        st.subheader(f"{day}")
        col_ac, col_wm, col_fridge = st.columns(3)
        with col_ac:
            ac_check = st.checkbox("AC", key=f"ac_{day}")
        with col_wm:
            wm_check = st.checkbox("Washing Machine", key=f"wm_{day}")
        with col_fridge:
            fridge_check = st.checkbox("Fridge", key=f"fridge_{day}")
        weekly_usage[day] = {
            'AC': ac_check,
            'Washing Machine': wm_check,
            'Fridge': fridge_check
        }

# Calculate energy
def calculate_energy():
    if facility == "1BHK":
        return 2 * 0.4 + 2 * 0.8
    elif facility == "2BHK":
        return 3 * 0.4 + 3 * 0.8
    elif facility == "3BHK":
        return 4 * 0.4 + 4 * 0.8
    return 0

# Calculate weekly usage
def calculate_weekly_usage():
    count = {'AC': 0, 'Washing Machine': 0, 'Fridge': 0}
    for usage in weekly_usage.values():
        for equip, used in usage.items():
            if used:
                count[equip] += 1
    return count

# Button click
if st.button("Calculate Consumption", type="primary"):
    if name and city and area:
        daily = calculate_energy()
        weekly = calculate_weekly_usage()
        base = daily

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Daily Consumption", f"{daily:.1f} kWh")
        with col2:
            monthly = daily * 30
            st.metric("Monthly Consumption", f"{monthly:.1f} kWh")
        with col3:
            st.metric("Estimated Monthly Cost", f"₹{monthly * 5:.0f}")

        # User Summary
        st.header("👤 User Summary")
        st.info(f"""
        **Name:** {name}  
        **Age:** {age}  
        **Location:** {area}, {city}  
        **Housing:** {facility} {flat_tenament}
        """)

        # Weekly Equipment Bar Chart
        st.header("📊 Weekly Equipment Usage Summary")
        usage_df = pd.DataFrame.from_dict(weekly, orient='index', columns=['Days Used'])
        st.bar_chart(usage_df)

        # Daily Equipment Table
        st.subheader("📅 Daily Equipment Usage Table")
        daily_table = []
        for day in days:
            row = {'Day': day}
            row.update(weekly_usage[day])
            daily_table.append(row)
        df_daily = pd.DataFrame(daily_table)
        st.table(df_daily)

        # Consumption Breakdown
        st.header("💡 Consumption Breakdown")
        breakdown = {
            'Base Consumption': base
        }
        breakdown_df = pd.DataFrame(list(breakdown.items()), columns=["Component", "kWh"])
        st.table(breakdown_df)

        # Weekly Summary Table
        st.header("📋 Weekly Summary")
        weekly_summary = []
        for day in days:
            daily_extra = sum(3 for eq, used in weekly_usage[day].items() if used)
            used_eq = [eq for eq, used in weekly_usage[day].items() if used]
            total = base + daily_extra
            weekly_summary.append({
                "Day": day,
                "Equipment Used": ", ".join(used_eq) if used_eq else "Base only",
                "Daily Consumption (kWh)": total,
                "Daily Cost (₹)": total * 5
            })

        df_summary = pd.DataFrame(weekly_summary)
        st.dataframe(df_summary, use_container_width=True)

        total_weekly = df_summary["Daily Consumption (kWh)"].sum()
        total_cost = df_summary["Daily Cost (₹)"].sum()
        avg_daily = total_weekly / 7

        st.success(f"""
        **Weekly Total:** {total_weekly:.1f} kWh  
        **Weekly Cost:** ₹{total_cost:.0f}  
        **Average Daily:** {avg_daily:.1f} kWh
        """)

    else:
        st.error("Please fill in all required fields (Name, City, Area)")

# Footer
st.markdown("---")
st.markdown("💡 **Note:** Energy consumption values are approximate. Actual consumption may vary based on usage patterns, appliance efficiency, and local conditions.")
st.markdown("💰 **Cost Calculation:** Based on average electricity rate of ₹5 per kWh.")
