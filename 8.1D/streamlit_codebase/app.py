import math
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Sydney Housing Price Predictor",
    layout="wide",
)


MODEL_PATH = "UI.joblib"

SUBURB_MEDIANS = {
    "Chippendale": 915000,
    "Parramatta": 691500,
    "Penrith": 656000,
}

PROPERTY_TYPES = [
    "Apartment",
    "House",
    "Townhouse",
    "Studio",
    "Duplex",
    "Semi-detached",
    "Block of units",
]

SALE_METHODS = [
    "Private Treaty",
    "Auction",
    "Prior to Auction",
]

FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "parking",
    "advertised_area_m2",
    "latitude",
    "longitude",
    "distance_to_cbd_km",
    "distance_to_nearest_station_km",
    "months_since_first_sale",
    "total_rooms",
    "has_parking",
    "suburb",
    "property_type",
    "sale_method",
]


@st.cache_resource
def load_model():
    # Load the fitted preprocessing + Random Forest pipeline once.
    return joblib.load(MODEL_PATH)


def optional_number(value):
    # Keep optional fields as missing when the user leaves them blank.
    value = value.strip()
    if value == "":
        return np.nan

    try:
        return float(value)
    except ValueError:
        return None


def months_since_reference(selected_date):
    # October 2025 is the earliest month represented in the dataset.
    return (
        (selected_date.year - 2025) * 12
        + (selected_date.month - 10)
    )


def build_input_row(
    suburb,
    property_type,
    bedrooms,
    bathrooms,
    parking,
    sale_method,
    sale_date,
    advertised_area_m2,
    latitude,
    longitude,
    distance_to_cbd_km,
    distance_to_nearest_station_km,
):
    # Recreate the engineered features used during model training.
    total_rooms = bedrooms + bathrooms
    has_parking = int(parking > 0)
    months_since_first_sale = months_since_reference(sale_date)

    row = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking": parking,
        "advertised_area_m2": advertised_area_m2,
        "latitude": latitude,
        "longitude": longitude,
        "distance_to_cbd_km": distance_to_cbd_km,
        "distance_to_nearest_station_km": distance_to_nearest_station_km,
        "months_since_first_sale": months_since_first_sale,
        "total_rooms": total_rooms,
        "has_parking": has_parking,
        "suburb": suburb,
        "property_type": property_type,
        "sale_method": sale_method,
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .app-title {
            font-size: 2.15rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }

        .app-subtitle {
            color: #5f6368;
            font-size: 1rem;
            margin-bottom: 1.6rem;
        }

        .result-label {
            color: #5f6368;
            font-size: 0.95rem;
            margin-bottom: 0.15rem;
        }

        .result-value {
            font-size: 2.5rem;
            font-weight: 750;
            line-height: 1.1;
            margin-bottom: 0.8rem;
        }

        .small-note {
            color: #666;
            font-size: 0.9rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #e6e6e6;
            border-radius: 10px;
            padding: 0.8rem 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Sydney Housing Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Estimate residential sale prices in Chippendale, Parramatta and Penrith using the final tuned Random Forest model.</div>',
    unsafe_allow_html=True,
)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "The model file could not be found. Place "
        "`housing_price_model.joblib` in the same folder as `app.py`."
    )
    st.stop()
except Exception as exc:
    st.error("The saved model could not be loaded. Please check the model file.")
    st.caption(f"Technical detail: {exc}")
    st.stop()


input_col, output_col = st.columns([1.08, 0.92], gap="large")

with input_col:
    st.subheader("Property Details")

    with st.form("prediction_form"):
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            suburb = st.selectbox(
                "Suburb",
                ["Chippendale", "Parramatta", "Penrith"],
            )

        with row1_col2:
            property_type = st.selectbox(
                "Property Type",
                PROPERTY_TYPES,
            )

        row2_col1, row2_col2, row2_col3 = st.columns(3)

        with row2_col1:
            bedrooms = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=20,
                value=2,
                step=1,
            )

        with row2_col2:
            bathrooms = st.number_input(
                "Bathrooms",
                min_value=0,
                max_value=10,
                value=1,
                step=1,
            )

        with row2_col3:
            parking = st.number_input(
                "Parking Spaces",
                min_value=0,
                max_value=12,
                value=1,
                step=1,
            )

        row3_col1, row3_col2 = st.columns(2)

        with row3_col1:
            sale_method = st.selectbox(
                "Sale Method",
                SALE_METHODS,
            )

        with row3_col2:
            sale_date = st.date_input(
                "Sale Date",
                value=date.today(),
            )

        advertised_area_text = st.text_input(
            "Advertised Area (m²) — optional",
            placeholder="e.g. 110",
        )

        with st.expander("Optional location information"):
            st.caption(
                "Leave these fields blank if they are unavailable. "
                "The saved preprocessing pipeline will handle missing values."
            )

            loc_col1, loc_col2 = st.columns(2)

            with loc_col1:
                latitude_text = st.text_input(
                    "Latitude",
                    placeholder="e.g. -33.884",
                )

                distance_cbd_text = st.text_input(
                    "Distance to Sydney CBD (km)",
                    placeholder="e.g. 2.0",
                )

            with loc_col2:
                longitude_text = st.text_input(
                    "Longitude",
                    placeholder="e.g. 151.201",
                )

                distance_station_text = st.text_input(
                    "Distance to nearest station (km)",
                    placeholder="e.g. 0.5",
                )

        submitted = st.form_submit_button(
            "Predict Sale Price",
            use_container_width=True,
            type="primary",
        )


with output_col:
    st.subheader("Prediction")

    if not submitted:
        st.info(
            "Enter the property details and select **Predict Sale Price** "
            "to generate an estimate."
        )

    if submitted:
        advertised_area_m2 = optional_number(advertised_area_text)
        latitude = optional_number(latitude_text)
        longitude = optional_number(longitude_text)
        distance_to_cbd_km = optional_number(distance_cbd_text)
        distance_to_nearest_station_km = optional_number(distance_station_text)

        optional_values = [
            advertised_area_m2,
            latitude,
            longitude,
            distance_to_cbd_km,
            distance_to_nearest_station_km,
        ]

        if any(value is None for value in optional_values):
            st.error(
                "One or more optional fields contain invalid numbers. "
                "Please enter a valid number or leave the field blank."
            )
        elif months_since_reference(sale_date) < 0:
            st.error(
                "Please choose a sale date from October 2025 onwards so it "
                "matches the period represented by the modelling data."
            )
        else:
            input_data = build_input_row(
                suburb=suburb,
                property_type=property_type,
                bedrooms=int(bedrooms),
                bathrooms=int(bathrooms),
                parking=int(parking),
                sale_method=sale_method,
                sale_date=sale_date,
                advertised_area_m2=advertised_area_m2,
                latitude=latitude,
                longitude=longitude,
                distance_to_cbd_km=distance_to_cbd_km,
                distance_to_nearest_station_km=distance_to_nearest_station_km,
            )

            try:
                predicted_price = float(model.predict(input_data)[0])
            except Exception as exc:
                st.error(
                    "The prediction could not be generated. "
                    "Please check the entered property information."
                )
                st.caption(f"Technical detail: {exc}")
            else:
                suburb_median = SUBURB_MEDIANS[suburb]
                difference = predicted_price - suburb_median
                difference_pct = (difference / suburb_median) * 100

                st.markdown(
                    '<div class="result-label">Estimated Sale Price</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="result-value">${predicted_price:,.0f}</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Generated using the tuned Random Forest regression model.")

                if difference >= 0:
                    st.write(
                        f"This estimate is **${difference:,.0f} "
                        f"({difference_pct:.1f}%) above** the {suburb} dataset median."
                    )
                else:
                    st.write(
                        f"This estimate is **${abs(difference):,.0f} "
                        f"({abs(difference_pct):.1f}%) below** the {suburb} dataset median."
                    )

                comparison_df = pd.DataFrame(
                    {
                        "Price ($)": [predicted_price, suburb_median]
                    },
                    index=["Predicted Price", f"{suburb} Median"],
                )

                st.markdown("#### Predicted Price vs Suburb Median")
                st.bar_chart(comparison_df, y="Price ($)", height=280)

                if property_type in {"Block of units", "Studio", "Semi-detached", "Duplex"}:
                    st.warning(
                        "This property type has relatively limited representation "
                        "in the modelling dataset, so the estimate should be interpreted cautiously."
                    )


st.divider()

st.subheader("Model Information")

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Validation MAE", "$165,260")

with metric_col2:
    st.metric("Validation RMSE", "$278,810")

with metric_col3:
    st.metric("Validation R²", "0.621")

st.caption(
    "The final model was selected after comparing Ridge Regression, "
    "Random Forest Regression and RBF-based Support Vector Regression "
    "using five-fold cross-validation."
)

st.warning(
    "This prediction is an estimate based on a relatively small historical dataset. "
    "Results should be interpreted cautiously for luxury, unusually large, rare or "
    "sparsely represented properties and when important property information is unavailable."
)
