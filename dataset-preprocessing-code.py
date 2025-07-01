import sys
from pathlib import Path

import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split  

# ---------------------------------------------------------------------------------------------------
# 1) استاندارسازی مقادیر گمشده در DataFrame
# ---------------------------------------------------------------------------------------------------
def standardize_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    مقادیر متنی رایج برای داده‌های گمشده را به np.nan تبدیل می‌کند.
    در همین حال، فاصله‌های ابتدای/انتهای رشته‌ها را همه‌جا حذف می‌کند.
    """
    print("\n🔄 شروع بخش 2: استاندارسازی مقادیر گمشده و حذف فاصله‌ها")

    # حذف فاصله‌های ابتدای/انتهای رشته‌ها در همه‌ی سلول‌ها
    df = df.replace(r"^\s+|\s+$", "", regex=True)

    # سپس استاندارسازی مقادیر گمشده
    nan_pattern = r'(?i)^(nan|na|n/a|null|none|missing|-|\.)?$'
    df = df.replace(nan_pattern, np.nan, regex=True)

    print("⚙️ فاصله‌ها حذف و مقادیر گمشده استانداردسازی شدند.")
    return df
# ---------------------------------------------------------------------------------------------------
# 2) بارگذاری و بررسی اولیه دیتاست
# ---------------------------------------------------------------------------------------------------
def load_and_inspect_data(file_path: str) -> pd.DataFrame:
    """
    بارگذاری فایل CSV و انجام بررسی‌های اولیه:
      1. بررسی وجود فایل
      2. استاندارسازی مقادیر گمشده
      3. پاک‌سازی نام ستون‌ها
      4. نمایش شکل، اطلاعات ساختار، آمار توصیفی، پیش‌نمایش و مقادیر گمشده
      5. لیست کردن متغیرهای عددی و دسته‌ای
    """
    print("\n🔄 شروع بخش 1: بارگذاری و بررسی اولیه دیتاست")
    path = Path(file_path)
    # 2.1 بررسی وجود فایل
    print("  ↳ بررسی وجود فایل")
    if not path.is_file():
        print(f"🟥 فایل پیدا نشد: {file_path}")
        sys.exit(1)

    # 2.2 بارگذاری دیتافریم
    print("  ↳ بارگذاری دیتافریم")
    df = pd.read_csv(path)
    print(f"✅ دیتاست '{path.name}' با موفقیت بارگذاری شد.")

    # 2.3 استاندارسازی مقادیر گمشده
    df = standardize_nan_values(df)

    # 2.4 پاک‌سازی نام ستون‌ها
    print("\n  ↳ پاک‌سازی نام ستون‌ها")
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"\s+", "_", regex=True)
    )
    print("⚙️ نام ستون‌ها پس از پاک‌سازی:", df.columns.tolist(), sep="\n")

    # 2.5 نمایش شکل دیتافریم
    print(
        "\n 🔢 شکل دیتافریم:",
        f"{df.shape[0]} سطر x {df.shape[1]} ستون",
        sep="\n"
    )

    # 2.6 نمایش اطلاعات کلی دیتافریم
    print("\n--- A1: اطلاعات کلی دیتافریم ---")
    df.info()

    # 2.7 آمار توصیفی متغیرهای عددی
    print("\n--- B1: آمار توصیفی (متغیرهای عددی) ---")
    print(df.describe())

    # 2.8 آمار متغیرهای دسته‌ای
    print("\n--- C1: آمار متغیرهای دسته‌ای ---")
    print(df.describe(include=['object', 'category']).T)

    # 2.9 پیش‌نمایش (5 سطر اول)
    print("\n--- D1: پیش‌نمایش (5 سطر اول) ---")
    print(df.head(5))

    # 2.10 تعداد مقادیر گمشده
    print("\n--- E1: تعداد مقادیر گمشده ---")
    missing = df.isna().sum()
    if missing.any():
        print(missing[missing > 0])
    else:
        print("ستونی بدون مقدار گمشده یافت نشد.")

    # 2.11 لیست متغیرها بر اساس نوع
    print("\n--- F1: گروه بندی ستون ها ---")
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(
        f"\nمتغیرهای عددی ({len(num_features)}):",
        f"{num_features}",
        sep="\n"
    )
    print(
        f"\nمتغیرهای غیرعددی ({len(cat_features)}):",
        f"{cat_features}",
        sep="\n"
    )
    return df

# ---------------------------------------------------------------------------------------------------
# 3) تحلیل و بصری‌سازی داده‌های پرت (Outliers)
# ---------------------------------------------------------------------------------------------------
def analyze_outliers(df: pd.DataFrame, columns: list):
    """
    فقط آمار داده‌های پرت را برای هر ستون عددی محاسبه و چاپ می‌کند.
    """
    print("\n--- تحلیل آماری داده‌های پرت ---")
    # انتخاب ستون‌های عددی معتبر
    valid_cols = [col for col in columns if col in df.columns]
    numeric_cols = df[valid_cols].select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        print("❌ هیچ ستون عددی معتبری برای تحلیل داده‌های پرت یافت نشد.")
        return

    print(f"📊 ستون‌های عددی برای تحلیل: {numeric_cols}")

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        median = df[col].median()
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers_lower = df[df[col] < lower_fence]
        outliers_upper = df[df[col] > upper_fence]
        total_outliers = len(outliers_lower) + len(outliers_upper)
        non_outliers = df[(df[col] >= lower_fence) & (df[col] <= upper_fence)]
        min_val = non_outliers[col].min()
        max_val = non_outliers[col].max()
        mean_val = df[col].mean()
        std_dev = df[col].std()
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()

        if skewness == 0:
            skew_interpret = "متقارن است"
        elif skewness > 0:
            skew_interpret = "چولگی راست دارد"
        else:
            skew_interpret = "چولگی چپ دارد"

        if kurtosis == 0:
            kurt_interpret = "کشیدگی نرمال"
        elif kurtosis > 0:
            kurt_interpret = "کشیدگی مثبت (دم سنگین)"
        else:
            kurt_interpret = "کشیدگی منفی (دم سبک)"

        stats_summary = (
            f"\n--- آمار ستون '{col}' ---\n"
            f"  - Q1 (چارک اول): {q1:,.2f}\n"
            f"  - Median (Q2 - میانه): {median:,.2f}\n"
            f"  - Q3 (چارک سوم): {q3:,.2f}\n"
            f"  - IQR (دامنه بین چارکی): {iqr:,.2f}\n"
            f"  - مرز پایین (Lower Fence): {lower_fence:,.2f}\n"
            f"  - مرز بالا (Upper Fence): {upper_fence:,.2f}\n"
            f"  - حداقل مقدار (در محدوده مجاز): {min_val:,.2f}\n"
            f"  - حداکثر مقدار (در محدوده مجاز): {max_val:,.2f}\n"
            f"  - تعداد داده‌های پرت: {total_outliers} (پایین: {len(outliers_lower)}, بالا: {len(outliers_upper)})\n"
            f"  - میانگین: {mean_val:,.2f}\n"
            f"  - انحراف معیار: {std_dev:,.2f}\n"
            f"  - چولگی (Skewness): {skewness:,.4f} ({skew_interpret})\n"
            f"  - کشیدگی (Kurtosis): {kurtosis:,.4f} ({kurt_interpret})"
        )
        print(stats_summary)


def visualize_outliers(df: pd.DataFrame, columns: list, save_base_name: str = "outlier_analysis"):
    """
    فقط نمودارهای Box Plot را می‌سازد و ذخیره می‌کند (بدون چاپ آمار).
    """
    print("\n--- رسم و ذخیره نمودارهای Box Plot برای داده‌های پرت ---")
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    valid_cols = [col for col in columns if col in df.columns]
    numeric_cols = df[valid_cols].select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        print("❌ هیچ ستون عددی معتبری برای نمودار داده‌های پرت یافت نشد.")
        return

    # بخش اول: Box Plot استاتیک با Matplotlib/Seaborn
    print("\n📈 ساخت نمودارهای استاتیک ...")
    plt.style.use('seaborn-v0_8-whitegrid')
    num_plots = len(numeric_cols)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), squeeze=False)
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[0, i], color="skyblue")
        axes[0, i].set_title(f'Box Plot for "{col}"', fontsize=14)
        axes[0, i].set_ylabel('مقدار', fontsize=12)
        axes[0, i].set_xlabel('')
    plt.tight_layout()
    plt.show()

    # بخش دوم: Box Plot تعاملی با Plotly
    print("\n🌐 ساخت نمودارهای تعاملی (HTML) ...")
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig_combined = make_subplots(rows=1, cols=len(numeric_cols), subplot_titles=numeric_cols)
    for i, col in enumerate(numeric_cols, 1):
        fig_combined.add_trace(go.Box(y=df[col], name=col, boxpoints='outliers', boxmean=True), row=1, col=i)
    fig_combined.update_layout(title_text="تحلیل داده‌های پرت - نمودار ترکیبی", showlegend=False)
    combined_file_path = os.path.join(output_dir, f"{save_base_name}_combined.html")
    fig_combined.write_html(combined_file_path)
    print(f"  - ✔️ نمودار ترکیبی در: {combined_file_path}")

    for col in numeric_cols:
        fig_single = go.Figure()
        fig_single.add_trace(go.Box(y=df[col], name=col, boxpoints='all', jitter=0.3, pointpos=-1.8))
        fig_single.update_layout(title=f'تحلیل داده‌های پرت برای "{col}"')
        single_file_path = os.path.join(output_dir, f"{save_base_name}_{col}.html")
        fig_single.write_html(single_file_path)
        print(f"  - ✔️ نمودار تکی ستون '{col}' در: {single_file_path}")

# ---------------------------------------------------------------------------------------------------
# 4) تبدیل لگاریتمی بر روی یک ستون داده‌ای
# ---------------------------------------------------------------------------------------------------
def visualize_distribution(df, column, bins='auto', kde=False, max_categories=50):
    """
    نمودار توزیع داده برای یک ستون:
     - اگر عددی باشد: هیستوگرام با گزینه KDE (تخمین چگالی)
     - اگر دسته‌ای باشد: نمودار شمارش (Bar chart)
    Args:
        df (pd.DataFrame): دیتافریم
        column (str): نام ستون
        bins (int or 'auto'): تعداد سطل‌های هیستوگرام
        kde (bool): نمایش KDE برای عددی
        max_categories (int): بیشینه تعداد دسته‌ برای نمایش (بارچارت)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if column not in df.columns:
        print(f"❌ ستون '{column}' یافت نشد.")
        return

    col_data = df[column].dropna()
    plt.figure(figsize=(10, 5))
    if pd.api.types.is_numeric_dtype(col_data):
        sns.histplot(col_data, bins=bins, kde=kde, color="skyblue")
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("تعداد")
    else:
        value_counts = col_data.value_counts()
        if len(value_counts) > max_categories:
            value_counts = value_counts.head(max_categories)
            plt.title(f"Bar chart of {column} (Top {max_categories})")
        else:
            plt.title(f"Bar chart of {column}")
        sns.barplot(x=value_counts.index.astype(str), y=value_counts.values, palette="viridis")
        plt.xlabel(column)
        plt.ylabel("تعداد")
        plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()


def log_transform(df: pd.DataFrame, column: str, new_column_name: str, constant: float = 1) -> pd.DataFrame:
    """
    تبدیل لگاریتمی بر روی یک ستون داده‌ای
    - لگاریتم طبیعی (ln) روی (column + constant)
    - مقدار پیش‌فرض constant=1 برای جلوگیری از log(0)
    - اگر مقادیر منفی یا صفر وجود داشته باشند، توصیه می‌شود constant بزرگتر تنظیم شود تا تمام مقادیر مثبت شوند

    Args:
        df (pd.DataFrame): دیتافریم ورودی
        column (str): نام ستون هدف
        new_column_name (str): نام ستون جدید
        constant (float): عدد ثابتی که به ستون اضافه می‌شود (پیش‌فرض 1)
        
    Returns:
        pd.DataFrame: دیتافریم با ستون جدید
    """
    print("\n---تبدیل لگاریتمی---")
    if column not in df.columns:
        raise ValueError(f"ستون '{column}' وجود ندارد.")

    transformed = np.log(df[column] + constant)
    df[new_column_name] = transformed
    print(f"⭐️ ستون لگاریتمی '{new_column_name}' به دیتافریم افزوده شد.")
    return df

# ---------------------------------------------------------------------------------------------------
# 5) label_encode
# ---------------------------------------------------------------------------------------------------
def label_encode_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    عمل Label Encoding را روی یک ستون دلخواه انجام داده و مقادیر عددی را جایگزین همان ستون می‌کند.
    مقادیر گمشده (NaN) به -1 نگاشت می‌شوند و نگاشت دسته‌ها چاپ می‌شود.

    پارامترها:
        df (pd.DataFrame): دیتافریم ورودی
        column (str): نام ستون دسته‌ای مورد نظر

    خروجی:
        df (pd.DataFrame): دیتافریم با ستون encode شده (در همان ستون)

    نمونه استفاده:
        label_encode_column(df_loans, "purpose")
    """

    print("\n---عملیات label_encode---")

    if column not in df.columns:
        print(f"❌ ستون '{column}' در دیتافریم وجود ندارد.")
        return df

    col_data = df[column].copy()
    encoded, uniques = pd.factorize(col_data)
    df[column] = encoded

    # نمایش نگاشت دسته‌ها به عدد
    print(f"🔢 Label Encoding برای ستون '{column}':")
    for idx, val in enumerate(uniques):
        print(f"  '{val}' → {idx}")

    if -1 in encoded:
        print("  ⚠️ توجه: مقادیر گمشده به -1 تبدیل شده‌اند")

    return df

# ---------------------------------------------------------------------------------------------------
#  6) اختلاف تعداد روز بین دو ستون تاریخ
# ---------------------------------------------------------------------------------------------------
def add_days_between_columns(df: pd.DataFrame, start_col: str, end_col: str, new_col_name: str) -> pd.DataFrame:
    """
    اختلاف تعداد روز بین دو ستون تاریخ را محاسبه کرده و به صورت یک ستون جدید به دیتافریم اضافه می‌کند.

    پارامترها:
        df (pd.DataFrame): دیتافریم ورودی
        start_col (str): نام ستون تاریخ شروع (مثلاً 'loan_start')
        end_col (str): نام ستون تاریخ پایان (مثلاً 'loan_end')
        new_col_name (str): نام ستون جدید خروجی (مثلاً 'days_diff')

    بازگشت:
        df (pd.DataFrame): دیتافریم با ستون جدید

    مثال:
        add_days_between_columns(df_loans, "loan_start", "loan_end", "duration_days")
    """
    print("\n---محاسبه اختلاف بین دو تاریخ---")

    if start_col not in df.columns or end_col not in df.columns:
        print(f"❌ یکی از ستون‌های '{start_col}' یا '{end_col}' وجود ندارد.")
        return df

    # تبدیل به تاریخ
    df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
    df[end_col] = pd.to_datetime(df[end_col], errors='coerce')

    # اختلاف روزها
    df[new_col_name] = (df[end_col] - df[start_col]).dt.days

    print(f"🟢 ستون جدید '{new_col_name}' (تعداد روز بین '{start_col}' و '{end_col}') به دیتافریم افزوده شد.")
    return df

# ---------------------------------------------------------------------------------------------------
#  7) ضرب دو ستون عددی
# ---------------------------------------------------------------------------------------------------
def multiply_columns(df: pd.DataFrame, col1: str, col2: str, new_col_name: str) -> pd.DataFrame:
    """
    ضرب دو ستون عددی و درج نتیجه در یک ستون جدید.

    پارامترها:
        df (pd.DataFrame): دیتافریم ورودی
        col1 (str): نام ستون اول
        col2 (str): نام ستون دوم
        new_col_name (str): نام ستون جدید

    بازگشت:
        df (pd.DataFrame): دیتافریم با ستون جدید

    مثال:
        multiply_columns(df_loans, "rate", "loan_amount", "total_payment")
    """
    print("\n---ضرب دو ستون---")

    if col1 not in df.columns or col2 not in df.columns:
        print(f"❌ یکی از ستون‌های '{col1}' یا '{col2}' یافت نشد.")
        return df

    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        print("❗️ هر دو ستون باید عددی باشند.")
        return df

    df[new_col_name] = df[col1] * df[col2]
    print(f"🟢 ستون جدید '{new_col_name}' (ضرب '{col1}' در '{col2}') ساخته شد.")
    return df

# ---------------------------------------------------------------------------------------------------
#  8) تقسیم دو ستون عددی
# ---------------------------------------------------------------------------------------------------
def divide_columns(df: pd.DataFrame, numerator_col: str, denominator_col: str, new_col_name: str) -> pd.DataFrame:
    """
    یک ستون جدید با نسبت (تقسیم) دو ستون عددی می‌سازد.
    اگر مخرج صفر یا NaN باشد نتیجه NaN خواهد بود و به کاربر هشدار داده می‌شود.

    پارامترها:
        df (pd.DataFrame): دیتافریم ورودی
        numerator_col (str): نام ستون صورت
        denominator_col (str): نام ستون مخرج
        new_col_name (str): نام ستون جدید

    بازگشت:
        df (pd.DataFrame): دیتافریم با ستون جدید

    مثال:
        divide_columns(df_loans, "rate_times_loan_amount", "duration_days", "rate_amount_per_day")
    """
    import pandas as pd
    import numpy as np

    if numerator_col not in df.columns or denominator_col not in df.columns:
        print(f"❌ یکی از ستون‌های '{numerator_col}' یا '{denominator_col}' یافت نشد.")
        return df

    bad_rows = df[denominator_col].isna() | (df[denominator_col] == 0)
    n_bad = bad_rows.sum()
    if n_bad > 0:
        print(f"⚠️ هشدار: {n_bad} سطر دارای مقدار صفر یا NaN در '{denominator_col}' هستند؛ مقدار بخش‌پذیری آنها NaN می‌شود.")

    df[new_col_name] = np.where(
        bad_rows,
        np.nan,
        df[numerator_col] / df[denominator_col]
    )
    print(f"🟢 ستون جدید '{new_col_name}' = '{numerator_col}' تقسیم بر '{denominator_col}' اضافه شد.")
    return df

# ---------------------------------------------------------------------------------------------------  
#  9) استانداردسازی ستون‌ها  
# ---------------------------------------------------------------------------------------------------  
def standardize_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    مقادیر ستون‌های مشخص شده را استانداردسازی می‌کند.
    
    پارامترها:
        df (pd.DataFrame): دیتافریم ورودی
        columns (list): لیست نام ستون‌هایی که باید استانداردسازی شوند
    
    بازگشت:
        df (pd.DataFrame): دیتافریم با ستون‌های استانداردسازی شده
    """
    print("\n---استانداردسازی ستون‌ها  ---")
    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std_dev = df[col].std()
            df[col] = (df[col] - mean) / std_dev
            print(f"🟢 ستون '{col}' با موفقیت استانداردسازی شد.")
        else:
            print(f"❌ ستون '{col}' در دیتافریم وجود ندارد.")
    return df

# ---------------------------------------------------------------------------------------------------
#  10) حذف ستون‌ها
# ---------------------------------------------------------------------------------------------------
def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    ستون‌های مشخص شده را از دیتافریم حذف می‌کند.
    
    پارامترها:
        df (pd.DataFrame): دیتافریم ورودی
        columns (list): لیست نام ستون‌هایی که باید حذف شوند
    
    بازگشت:
        df (pd.DataFrame): دیتافریم با ستون‌های حذف شده
    """
    print("\n--- حذف ستون‌ها  ---")
    for col in columns:
        if col in df.columns:
            df = df.drop(col, axis=1)
            print(f"🟢 ستون '{col}' با موفقیت حذف شد.")
        else:
            print(f"❌ ستون '{col}' در دیتافریم وجود ندارد.")
    return df


# ---------------------------------------------------------------------------------------------------
#  10) ذخیره دیتافریم به CSV
# ---------------------------------------------------------------------------------------------------
def save_dataframe(df, output_dir="data", filename="processed_data", index=False):
    """
    دیتافریم را به صورت CSV در پوشه مشخص‌شده و با نام دلخواه ذخیره می‌کند.
    
    پارامترها:
        df (pd.DataFrame): دیتافریم موردنظر
        output_dir (str): نام پوشه خروجی (پیش فرض "data")
        filename (str): نام فایل خروجی بدون پسوند (پیش فرض "processed_data")
        index (bool): ذخیره اندیس سطرها یا خیر (پیش فرض False)
    """
    print("\n---ذخیره دیتافریم---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(output_path, index=index, encoding="utf-8-sig")
    print(f"💾 دیتافریم با موفقیت در '{output_path}' ذخیره شد.")

# ---------------------------------------------------------------------------------------------------
#  11) تقسیم دیتافریم به مجموعه‌های آموزش و آزمون
# ---------------------------------------------------------------------------------------------------
def split_and_save_train_test(df, test_size=0.2, random_state=42, 
                              output_dir="data", train_name="train_set", test_name="test_set", index=False):
    """
    دیتافریم را به دو بخش آموزش و آزمون تقسیم کرده و ذخیره می‌کند.
    
    پارامترها:
        df : دیتافریم ورودی
        test_size : نسبت آزمون (بین ۰ و ۱)
        random_state : عدد ثابت برای تکرارپذیری
        output_dir : پوشه ذخیره  
        train_name : نام فایل بخش آموزش  
        test_name : نام فایل بخش آزمون  
        index : آیا ایندکس ذخیره شود یا نه
    
    بازگشت:
        df_train, df_test
    """
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    save_dataframe(df_train, output_dir=output_dir, filename=train_name, index=index)
    save_dataframe(df_test, output_dir=output_dir, filename=test_name, index=index)
    print(f"✅ تقسیم و ذخیره‌سازی مجموعه‌های آموزش و آزمون انجام شد.")
    return df_train, df_test

# ---------------------------------------------------------------------------------------------------
#  اجرای اصلی برنامه
# ---------------------------------------------------------------------------------------------------

csv_path = "./data/loans.csv"
# بارگذاری و بررسی اولیه دیتاست
df_loans = load_and_inspect_data(csv_path)

# 2. فراخوانی تابع تحلیل داده‌های پرت
# لیستی از ستون‌های عددی مورد نظر خود را اینجا وارد کنید
columns_for_outlier_analysis = ["loan_amount", "rate"] 

# 2.1. فقط تحلیل آماری داده‌های پرت
analyze_outliers(df_loans, columns_for_outlier_analysis)

# 2.2. فقط رسم و ذخیره نمودارهای داده‌های پرت
visualize_outliers(df_loans, columns_for_outlier_analysis, save_base_name="outlier_analysis")

# 3. تبدیل لگاریتمی بر روی یک ستون
# visualize_distribution(df_loans, "rate")
df_loans = log_transform(df_loans, column="rate", new_column_name="log_rate")

# 4. Label Encoding 
column_to_encode = "loan_type"  
df_loans = label_encode_column(df_loans, column_to_encode)

# 5. ایجاد ویژگی‌های جدید
df_loans = add_days_between_columns(df_loans, "loan_start", "loan_end", "duration_days")
df_loans = multiply_columns(df_loans, "rate", "loan_amount", "rate_times_loan_amount")
df_loans = divide_columns(df_loans, "rate_times_loan_amount", "duration_days", "amount_per_day")

# 6. استانداردسازی ستون‌ها
# لیست ستون‌هایی که باید استانداردسازی شوند
columns_to_standardize = ["loan_amount", "log_rate", "duration_days", "rate_times_loan_amount", "amount_per_day"]
df_loans = standardize_columns(df_loans, columns_to_standardize)

# 7. حذف ستون‌ها
# لیست ستون‌هایی که باید حذف شوند
columns_to_drop = ["rate", "loan_id"]
df_loans = drop_columns(df_loans, columns_to_drop)



# 8. تقسیم و ذخیره‌سازی دیتافریم به مجموعه‌های آموزش و آزمون
df_train, df_test = split_and_save_train_test(df_loans, test_size=0.2, random_state=42)


# 9. ذخیره دیتافریم نهایی
save_dataframe(df_loans, output_dir="data", filename="loans_final", index=False)