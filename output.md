<div dir="rtl">



<h4>🔄 شروع بخش 1: بارگذاری و بررسی اولیه دیتاست</h4>
<ul>
    <li>↳ بررسی وجود فایل</li>
    <li>↳ بارگذاری دیتافریم</li>
</ul>

<p>✅ دیتاست 'loans1.csv' با موفقیت بارگذاری شد.</p>

<hr>

<h4>🔄 شروع بخش 2: استاندارسازی مقادیر گمشده و حذف فاصله‌ها</h4>
<p>⚙️ فاصله‌ها حذف و مقادیر گمشده استانداردسازی شدند.</p>

<ul>
    <li>↳ پاک‌سازی نام ستون‌ها</li>
    <li>⚙️ نام ستون‌ها پس از پاک‌سازی:</li>
</ul>
<pre><code>['client_id', 'loan_type', 'loan_amount', 'repaid', 'loan_id', 'loan_start', 'loan_end', 'rate']</code></pre>

<hr>

<h4>🔢 شکل دیتافریم:</h4>
<pre><code>443 سطر x 8 ستون</code></pre>

<hr>

<h4 dir="rtl">--- A1: اطلاعات کلی دیتافریم ---</h4>

<p>&lt;class 'pandas.core.frame.DataFrame'&gt;<br>
RangeIndex: 443 entries, 0 to 442<br>
Data columns (total 8 columns):</p>
<div align=left>
<table border="1" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;">
  <thead>
    <tr style="background-color: #4CAF50; color: white;">
      <th style="padding: 8px; text-align: center;">#</th>
      <th style="padding: 8px; text-align: left;">Column</th>
      <th style="padding: 8px; text-align: center;">Non-Null Count</th>
      <th style="padding: 8px; text-align: center;">Dtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">0</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">client_id</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">443 non-null</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">int64</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">1</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">loan_type</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">443 non-null</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">object</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">2</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">loan_amount</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">443 non-null</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">int64</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">3</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">repaid</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">443 non-null</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">int64</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">4</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">loan_id</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">443 non-null</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">int64</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">5</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">loan_start</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">443 non-null</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">object</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">6</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">loan_end</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">443 non-null</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">object</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">7</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">rate</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">443 non-null</td>
      <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">float64</td>
    </tr>
  </tbody>
</table>
</div>
<p>dtypes: float64(1), int64(4), object(3)<br>
memory usage: 27.8+ KB</p>

<hr>

<h4 dir="rtl">--- B1: آمار توصیفی (متغیرهای عددی) ---</h4>
<div align=center>
<table border="1" style="border-collapse: collapse; width: 100%;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th></th>
      <th>client_id</th>
      <th>loan_amount</th>
      <th>repaid</th>
      <th>loan_id</th>
      <th>rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="font-weight: bold;">count</td>
      <td>443.000000</td>
      <td>443.000000</td>
      <td>443.000000</td>
      <td>443.000000</td>
      <td>443.000000</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">mean</td>
      <td>38911.060948</td>
      <td>7982.311512</td>
      <td>0.534989</td>
      <td>11017.101580</td>
      <td>3.217156</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">std</td>
      <td>7768.681063</td>
      <td>4172.891992</td>
      <td>0.499338</td>
      <td>581.826222</td>
      <td>2.397168</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">min</td>
      <td>25707.000000</td>
      <td>559.000000</td>
      <td>0.000000</td>
      <td>10009.000000</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">25%</td>
      <td>32885.000000</td>
      <td>4232.500000</td>
      <td>0.000000</td>
      <td>10507.500000</td>
      <td>1.220000</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">50%</td>
      <td>39505.000000</td>
      <td>8320.000000</td>
      <td>1.000000</td>
      <td>11033.000000</td>
      <td>2.780000</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">75%</td>
      <td>46109.000000</td>
      <td>11739.000000</td>
      <td>1.000000</td>
      <td>11526.000000</td>
      <td>4.750000</td>
    </tr>
    <tr>
      <td style="font-weight: bold;">max</td>
      <td>49624.000000</td>
      <td>14971.000000</td>
      <td>1.000000</td>
      <td>11991.000000</td>
      <td>12.620000</td>
    </tr>
  </tbody>
</table>
</div>
<hr>

<h4>--- C1: آمار متغیرهای دسته‌ای ---</h4>
<div align=center>
<table border="1" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; margin-bottom: 20px;">
  <thead>
    <tr style="background-color: #2196F3; color: white;">
      <th style="padding: 10px; text-align: center;"></th>
      <th style="padding: 10px; text-align: center;">count</th>
      <th style="padding: 10px; text-align: center;">unique</th>
      <th style="padding: 10px; text-align: center;">top</th>
      <th style="padding: 10px; text-align: center;">freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 10px; font-weight: bold; border: 1px solid #ddd;">loan_type</td>
      <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">443</td>
      <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">4</td>
      <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">home</td>
      <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">121</td>
    </tr>
    <tr>
      <td style="padding: 10px; font-weight: bold; border: 1px solid #ddd;">loan_start</td>
      <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">443</td>
      <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">430</td>
      <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">2007-05-16</td>
      <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">2</td>
    </tr>
    <tr>
      <td style="padding: 10px; font-weight: bold; border: 1px solid #ddd;">loan_end</td>
      <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">443</td>
      <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">428</td>
      <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">2008-08-29</td>
      <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">2</td>
    </tr>
  </tbody>
</table>
</div>

<hr>

<h4>--- D1: پیش‌نمایش (5 سطر اول) ---</h4>
<div align=center>
<table border="1" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;">
  <thead>
    <tr style="background-color: #FF9800; color: white;">
      <th style="padding: 10px; text-align: center;"></th>
      <th style="padding: 10px; text-align: center;">client_id</th>
      <th style="padding: 10px; text-align: center;">loan_type</th>
      <th style="padding: 10px; text-align: center;">loan_amount</th>
      <th style="padding: 10px; text-align: center;">repaid</th>
      <th style="padding: 10px; text-align: center;">loan_id</th>
      <th style="padding: 10px; text-align: center;">loan_start</th>
      <th style="padding: 10px; text-align: center;">loan_end</th>
      <th style="padding: 10px; text-align: center;">rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td >0</td>
      <td >46109</td>
      <td >home</td>
      <td >13672</td>
      <td>0</td>
      <td >10243</td>
      <td>2002-04-16</td>
      <td >2003-12-20</td>
      <td >2.15</td>
    </tr>
    <tr>
      <td >1</td>
      <td >46109</td>
      <td >credit</td>
      <td >9794</td>
      <td>0</td>
      <td >10984</td>
      <td >2003-10-21</td>
      <td >2005-07-17</td>
      <td >1.25</td>
    </tr>
    <tr>
      <td>2</td>
      <td >46109</td>
      <td >home</td>
      <td >12734</td>
      <td >1</td>
      <td>10990</td>
      <td >2006-02-01</td>
      <td >2007-07-05</td>
      <td >0.68</td>
    </tr>
    <tr>
      <td >3</td>
      <td >46109</td>
      <td >cash</td>
      <td >12518</td>
      <td>1</td>
      <td >10596</td>
      <td >2010-12-08</td>
      <td >2013-05-05</td>
      <td >1.24</td>
    </tr>
    <tr>
      <td >4</td>
      <td>46109</td>
      <td>credit</td>
      <td >14049</td>
      <td >1</td>
      <td >11415</td>
      <td >2010-07-07</td>
      <td >2012-05-21</td>
      <td>3.13</td>
    </tr>
  </tbody>
</table>
</div>

<hr>

<h4 dir="rtl">--- E1: تعداد مقادیر گمشده ---</h4>
<p>ستونی بدون مقدار گمشده یافت نشد.</p>

<hr>

<h4 dir="rtl">--- F1: گروه بندی ستون ها ---</h4>
<p>
متغیرهای عددی (5):<br>
<code>['client_id', 'loan_amount', 'repaid', 'loan_id', 'rate']</code><br><br>
متغیرهای غیرعددی (3):<br>
<code>['loan_type', 'loan_start', 'loan_end']</code>
</p>

<hr>

<h4 dir="rtl">--- آمار ستون 'loan_amount' ---</h4>
<ul dir="rtl">
  <li>Q1 (چارک اول): 4,232.50</li>
  <li>Median (Q2 - میانه): 8,320.00</li>
  <li>Q3 (چارک سوم): 11,739.00</li>
  <li>IQR (دامنه بین چارکی): 7,506.50</li>
  <li>مرز پایین (Lower Fence): -7,027.25</li>
  <li>مرز بالا (Upper Fence): 22,998.75</li>
  <li>حداقل مقدار (در محدوده مجاز): 559.00</li>
  <li>حداکثر مقدار (در محدوده مجاز): 14,971.00</li>
  <li>تعداد داده‌های پرت: 0 (پایین: 0, بالا: 0)</li>
  <li>میانگین: 7,982.31</li>
  <li>انحراف معیار: 4,172.89</li>
  <li>چولگی (Skewness): -0.0401 (چولگی چپ دارد)</li>
  <li>کشیدگی (Kurtosis): -1.2321 (کشیدگی منفی (دم سبک))</li>
</ul>

<hr>

<h4 dir="rtl">--- آمار ستون 'rate' ---</h4>
<ul dir="rtl">
  <li>Q1 (چارک اول): 1.22</li>
  <li>Median (Q2 - میانه): 2.78</li>
  <li>Q3 (چارک سوم): 4.75</li>
  <li>IQR (دامنه بین چارکی): 3.53</li>
  <li>مرز پایین (Lower Fence): -4.08</li>
  <li>مرز بالا (Upper Fence): 10.04</li>
  <li>حداقل مقدار (در محدوده مجاز): 0.01</li>
  <li>حداکثر مقدار (در محدوده مجاز): 9.91</li>
  <li>تعداد داده‌های پرت: 3 (پایین: 0, بالا: 3)</li>
  <li>میانگین: 3.22</li>
  <li>انحراف معیار: 2.40</li>
  <li>چولگی (Skewness): 0.8842 (چولگی راست دارد)</li>
  <li>کشیدگی (Kurtosis): 0.4244 (کشیدگی مثبت (دم سنگین))</li>
</ul>

<hr>

<h4 dir="rtl">--- تبدیل لگاریتمی ---</h4>
<p>⭐️ ستون لگاریتمی 'log_rate' به دیتافریم افزوده شد.</p>

<hr>

<h4 dir="rtl">--- عملیات label_encode ---</h4>
<p>🔢 Label Encoding برای ستون 'loan_type':<br>
  'home' → 0<br>
  'credit' → 1<br>
  'cash' → 2<br>
  'other' → 3
</p>

<hr>

<h4 dir="rtl">--- محاسبه اختلاف بین دو تاریخ ---</h4>
<p>🟢 ستون جدید 'duration_days' (تعداد روز بین 'loan_start' و 'loan_end') به دیتافریم افزوده شد.</p>

<hr>

<h4 dir="rtl">--- ضرب دو ستون ---</h4>
<p>🟢 ستون جدید 'rate_times_loan_amount' (ضرب 'rate' در 'loan_amount') ساخته شد.<br>
🟢 ستون جدید 'amount_per_day' = 'rate_times_loan_amount' تقسیم بر 'duration_days' اضافه شد.</p>

<hr>

<h4 dir="rtl">--- استانداردسازی ستون‌ها ---</h4>
<p>🟢 ستون 'loan_amount' با موفقیت استانداردسازی شد.<br>
🟢 ستون 'log_rate' با موفقیت استانداردسازی شد.<br>
🟢 ستون 'duration_days' با موفقیت استانداردسازی شد.<br>
🟢 ستون 'rate_times_loan_amount' با موفقیت استانداردسازی شد.<br>
🟢 ستون 'amount_per_day' با موفقیت استانداردسازی شد.</p>

<hr>

<h4 dir="rtl">--- حذف ستون‌ها ---</h4>
<p>🟢 ستون 'rate' با موفقیت حذف شد.<br>
🟢 ستون 'loan_id' با موفقیت حذف شد.</p>

<hr>

<h4 dir="rtl">--- ذخیره دیتافریم ---</h4>
<p>💾 دیتافریم با موفقیت در 'data\train_set.csv' ذخیره شد.</p>

<hr>

<h4 dir="rtl">--- ذخیره دیتافریم ---</h4>
<p>💾 دیتافریم با موفقیت در 'data\test_set.csv' ذخیره شد.<br>
✅ تقسیم و ذخیره‌سازی مجموعه‌های آموزش و آزمون انجام شد.</p>

<hr>

<h4 dir="rtl">--- ذخیره دیتافریم ---</h4>
<p>💾 دیتافریم با موفقیت در 'data\loans_final2.csv' ذخیره شد.</p>

</div>
