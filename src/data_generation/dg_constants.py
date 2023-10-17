from pathlib import Path


class DataLocations:
    MARKETING_REVENUE = Path(
        "/Users/jakob/Documents/Erasmus/Eco/Master/Thesis/LATAM data/DATASET_MMM_v2.csv"
    )
    ASK_DOM = Path(
        "/Users/jakob/Documents/Erasmus/Eco/Master/Thesis/LATAM data/ask_DOM_chile.csv"
    )
    ASK_REG = Path(
        "/Users/jakob/Documents/Erasmus/Eco/Master/Thesis/LATAM data/ask_REG_LH_to_chile.csv"
    )
    PROMO = Path(
        "/Users/jakob/Documents/Erasmus/Eco/Master/Thesis/LATAM data/promo_dummies/promo_dummy_1day.csv"
    )
    FAKE_MARKETING = Path("src/data_generation/fake_marketing.csv")


class DataHelpers:
    """Removed: "S_DISPLAY_METASEARCH_MKT_AON_UPPER", "S_DV360_DISPLAY_PERF_LOWER", "S_CRITEO_DISPLAY_MKT_UPPER"
    Since they have no values in the training set

    low betas (<10000):
        i13:
            4 (S_FB_DISPLAY_PERF_LOWER)
            8 (S_GOOGLE_DISPLAY_CORP_BRANDING)
            10 (S_DV360_DISPLAY_MKT_UPPER)
            11 (S_DV360_DISPLAY_CORP_BRANDING)
            12 (S_RTBHOUSE_DISPLAY_MKT_UPPER)
        i14
            11 (S_DV360_DISPLAY_CORP_BRANDING)
            12 (S_RTBHOUSE_DISPLAY_MKT_UPPER)
            18 (S_RADIO_BRANDING)
        i15
            8 (S_GOOGLE_DISPLAY_CORP_BRANDING)
            11 (S_DV360_DISPLAY_CORP_BRANDING)
            17 (S_OOH_BRANDING)
            18 (S_RADIO_BRANDING)
        i17
            11 (S_DV360_DISPLAY_CORP_BRANDING)
            12 (S_RTBHOUSE_DISPLAY_MKT_UPPER)
            14 (S_CRITEO_DISPLAY_PERF_LOWER)
            17 (S_OOH_BRANDING)
            18 (S_RADIO_BRANDING)
        i19
            8 (S_GOOGLE_DISPLAY_CORP_BRANDING)
            17 (S_OOH_BRANDING)
        i22
            8 (S_GOOGLE_DISPLAY_CORP_BRANDING)
            11 (S_DV360_DISPLAY_CORP_BRANDING)
            14 (S_CRITEO_DISPLAY_PERF_LOWER)
            17 (S_OOH_BRANDING)


    Removed: "S_OTHER"
    Since is more like noise
    """

    USEFUL_CHANNELS = [
        "DATE",
        "REVENUE",
        "S_SEM_BRAND_AON_LOWER",
        "S_SEM_NON_BRAND_AON_LOWER",
        "S_FB_DISPLAY_MKT_UPPER",
        "S_FB_DISPLAY_PERF_LOWER",
        "S_FB_DISPLAY_CORP_BRANDING",
        "S_GOOGLE_DISPLAY_MKT_UPPER",
        "S_GOOGLE_DISPLAY_PERF_LOWER",
        # "S_GOOGLE_DISPLAY_CORP_BRANDING", Removed for test
        # "S_DISPLAY_METASEARCH_MKT_AON_UPPER", empty
        "S_DISPLAY_METASEARCH_PERF_AON_LOWER",
        "S_DV360_DISPLAY_MKT_UPPER",
        # "S_DV360_DISPLAY_PERF_LOWER", empty
        # "S_DV360_DISPLAY_CORP_BRANDING", removed for test
        "S_RTBHOUSE_DISPLAY_MKT_UPPER",
        "S_RTBHOUSE_DISPLAY_PERF_LOWER",
        # "S_CRITEO_DISPLAY_MKT_UPPER", empty
        "S_CRITEO_DISPLAY_PERF_LOWER",
        "S_METASEARCH_CORE_AON_LOWER",
        "S_TV_BRANDING",
        "S_OOH_BRANDING",
        "S_RADIO_BRANDING",
        # "S_OTHER", noise
    ]

    FAKE_CHANNELS = [
        "Channel_1",
        "Channel_2",
        "Channel_3",
        "Channel_4",
        "Channel_5",
        "Channel_6",
        "Channel_7",
        "Channel_8",
    ]
    START_DATE = "01/01/2019"
    END_DATE = "31/12/2022"

    ROLLING_WINDOW = 21
