<div dir="rtl">

<h4>🔢 اطلاعات کلی دیتاست:</h4>
<ul>
    <li>تعداد ردیف‌ها: 443</li>
    <li>تعداد ستون‌ها: 8</li>
    <li>ستون‌ها و نوع داده‌ها:</li>
    <ul>
        <li>عددی (۵ ستون):<br>
            <code>client_id (int64), loan_amount (int64), repaid (int64), loan_id (int64), rate (float64)</code>
        </li>
        <li>غیرعددی (۳ ستون):<br>
            <code>loan_type (object), loan_start (object), loan_end (object)</code>
        </li>
    </ul>
    <li>هیچ مقدار گمشده‌ای در دیتاست وجود ندارد.</li>
    <li>نام ستون‌ها پس از پاک‌سازی:<br>
        <code>['client_id', 'loan_type', 'loan_amount', 'repaid', 'loan_id', 'loan_start', 'loan_end', 'rate']</code>
    </li>
</ul>

<hr>

<h4>🔢 آمار توصیفی متغیرهای عددی:</h4>
<div align=center>
<table border="1" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;" dir="ltr">
    <thead>
        <tr style="background-color: #f2f2f2;">
            <th>متغیر</th>
            <th>count</th>
            <th>mean</th>
            <th>std</th>
            <th>min</th>
            <th>25%</th>
            <th>50%</th>
            <th>75%</th>
            <th>max</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>loan_amount</td>
            <td>443</td>
            <td>7,982.31</td>
            <td>4,172.89</td>
            <td>559</td>
            <td>4,232.50</td>
            <td>8,320.00</td>
            <td>11,739.00</td>
            <td>14,971</td>
        </tr>
        <tr>
            <td>repaid</td>
            <td>443</td>
            <td>0.535</td>
            <td>0.499</td>
            <td>0</td>
            <td>0</td>
            <td>1</td>
            <td>1</td>
            <td>1</td>
        </tr>
        <tr>
            <td>rate</td>
            <td>443</td>
            <td>3.217</td>
            <td>2.397</td>
            <td>0.01</td>
            <td>1.22</td>
            <td>2.78</td>
            <td>4.75</td>
            <td>12.62</td>
        </tr>
    </tbody>
</table></div>

<ul>
    <li>میزان وام (loan_amount):<br>
        •	متوسط حدود 8,000<br>
        •	پراکندگی بالا (انحراف معیار)
    </li>
    <li>وضعیت بازپرداخت (repaid):<br>
        •	میانگین 0.535 نشان می‌دهد که حدود 53.5٪ وام‌ها بازپرداخت شده‌اند.
    </li>
    <li>نرخ بهره (rate):<br>
        •	متوسط  3.217٪<br>
        •	دامنه از حداقل 0.01٪ تا سقف 12.62٪<br>
        •	انحراف معیار حدود 2.4 نشان‌دهنده تنوع نسبتاً زیاد در نرخ‌هاست.
    </li>
</ul>

<hr>

<h4>🔢 آمار متغیرهای دسته‌ای:</h4>
<div align=center>
<table border="1" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;"dir="ltr">
    <thead>
        <tr style="background-color: #f2f2f2;">
            <th>ستون</th>
            <th>تعداد یکتا</th>
            <th>پر تکرارترین مقدار</th>
            <th>فراوانی</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>loan_type</td>
            <td>4</td>
            <td>home</td>
            <td>121</td>
        </tr>
        <tr>
            <td>loan_start</td>
            <td>430</td>
            <td>2007-05-16</td>
            <td>2</td>
        </tr>
        <tr>
            <td>loan_end</td>
            <td>428</td>
            <td>2008-08-29</td>
            <td>2</td>
        </tr>
    </tbody>
</table></div>

<ul>
    <li>تقریباً همه تاریخ‌ها منحصر به فرد هستند (بیش از 400 تاریخ برای 443 رکورد)</li>
    <li>نوع وام (loan_type):<br>
        •	چهار گروه مختلف<br>
        •	بیشترین تعداد مربوط به «home» با 121 رکورد
    </li>
</ul>


<hr>

<h4>🔢 تحلیل داده‌های پرت (Outliers)</h4>


<ul>
    <li>تحلیل دو ستون عددیِ loan_amount و rate</li>
</ul>

<h3>تحلیل ستون loan_amount (مبلغ وام)</h3>

<ol  dir="rtl">
    <li><h4>مقادیر واقعی و پرت‌ها:</h4>
        <ul >
            <li  dir="rtl">Min = 559.00, Max = 14,971.00</li>
            <li>هیچ مقداری خارج از بازه $\left[-7,027.25\ ,\ 22,998.75\ \right]$ نیست در نتیجه مشخص است که داده پرتی نداریم.</li>
        </ul>
    </li>
    <li><h4>آمار توصیفی:</h4>
        <ul>
            <li>میانگین = 7,982.31</li>
            <li>انحراف معیار = 4,172.89</li>
            <li>چولگی (Skewness) برابر $-0.04$ است که نشان می‌دهد داده‌ها دارای توزیع تقریبا متقارن هستند.</li>
            <li>برجستگی (Kurtosis) برابر $-1.23$ که نشان می‌دهد میزان برجستگی کمتر از حالت نرمال است.</li>
        </ul>
    </li>
    <li><h4>تفسیر نهایی:</h4>
        <ul>
            <li>توزیع مبلغ وام‌ها سالم و بدون ناهنجاری است.</li>
            <li>نزدیکی میانگین و میانه و چولگی نزدیک صفر است که نشان می‌دهد داده‌ها تقریباً متقارن هستند.</li>
            <li>دامنه مبلغ وام‌ها فاقد مقادیر مشکوک است.</li>
        </ul>
    </li>
</ol>

<h3>تحلیل جامع ستون rate (نرخ بهره)</h3>
<ol  dir="rtl">
    <li><h4>مقادیر واقعی و پرت‌ها:</h4>
        <ul  dir="rtl">
            <li>Min = 0.01 (داخل بازه)</li>
            <li>Max = 12.62</li>
<li>
    تعداد مقادیری که از 10.04 بیشتر هستند برابر 3 است؛ پس می‌شود گفت سه داده پرت داریم 
    <span ><h3>
        (این سه نرخ بهره می‌توانند برای وام‌های پرخطر یا دارای شرایط ویژه باشند؛ پس آن‌ها را به عنوان داده پرت در نظر نمی‌گیریم و از داده‌ها حذف نمی‌کنیم).
    </h3>h3> </span>
</li>
        </ul>
    </li>
    <li><h4>آمار توصیفی کلی:</h4>
        <ul >
            <li dir="rtl">میانگین = 3.22</li>
            <li>انحراف معیار = 2.40</li>
            <li>چولگی (Skewness) برابر $+0.88$ است که نشان‌دهنده چولگی راست (مثبت) است که می‌توان از آن نتیجه گرفت که تمرکز بیشتر نرخ‌های بهره در مقادیر پایین است.</li>
            <li>برجستگی (Kurtosis) برابر $+0.42$ که نشان می‌دهد میزان برجستگی بیشتر از حالت نرمال است.</li>
        </ul>
    </li>
    <li><h4>تفسیر نهایی:</h4>
        <ul>
            <li>بیشتر نرخ‌های بهره حول میانهٔ 2.78٪ متمرکزند، اما چند وام با نرخ بسیار بالا (بالاتر از 10.04٪) وجود دارد.</li>
            <li>این نقاط پرت (۳ رکورد) می‌توانند وام‌های پرخطر یا دارای شرایط ویژه باشند.</li>
        </ul>
    </li>
</ol>

<hr>
<h4>🔄 تبدیل لگاریتمی بر روی ستون‌های داده‌ها:</h4>

<ul dir="rtl">
    <li dir="rtl">
        <h4  >برای ستون loan_amount:</h4>
        <img src="https://github.com/bahmanrakk/darsman-5-ML-1-Dataset-preprocessing/blob/main/data/images/loan_amount.png" 
             alt="Loan Amount Distribution" 
             style="max-width: 100%; height: auto;">
        <p>با توجه به اینکه مقدار چولگی برای این ستون (برابر $-0.0401$) و نزدیک صفر است، بهتر است تبدیلی برای آن انجام نشود.</p>
    </li>
    <li>
        <h4>برای ستون rate:</h4>
        <img src="https://github.com/bahmanrakk/darsman-5-ML-1-Dataset-preprocessing/blob/main/data/images/rate.png" 
             alt="Rate Distribution" 
             style="max-width: 100%; height: auto;">
        <p>چولگی ستون "rate" مثبت است ($0.8842$). این به این معنی است که اکثر داده‌ها در سمت چپ توزیع قرار دارند و چند مقدار بزرگ در سمت راست وجود دارند. تبدیل لگاریتمی می‌تواند کمک کند تا داده‌ها به توزیع نرمال نزدیک‌تر شوند.</p>
    </li>
</ul>
<hr>

<h4 dir="rtl">🔢 Label Encoding</h4>
<p>همانطور که در جدول بخش آمار متغیرهای دسته‌ای قابل مشاهده است:</p>
<div align=center>
<table border="1" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;" dir="ltr">
    <thead>
        <tr style="background-color: #f2f2f2;">
            <th>ستون</th>
            <th>تعداد یکتا</th>
            <th>پر تکرارترین مقدار</th>
            <th>فراوانی</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>loan_type</td>
            <td>4</td>
            <td>home</td>
            <td>121</td>
        </tr>
    </tbody>
</table></div>
<p>ستون loan_type دارای چهار مقدار برای همه عناصر است که می‌توان برای آن از روش Label Encoding استفاده کرد:</p>

<ul>
    <li><strong>'home'</strong> → 0</li>
    <li><strong>'credit'</strong> → 1</li>
    <li><strong>'cash'</strong> → 2</li>
    <li><strong>'other'</strong> → 3</li>
</ul>
</div>







<hr> 
<div dir="rtl">
<h4>🔄 ایجاد ویژگی‌های جدید</h4>
<ul  dir="rtl">
    <li><h5>: <code>duration_days</code> (مدت زمان وام به روز)</h5>
        <ul>
            <li>  از تفاضل تاریخ‌های <code>loan_start</code> و <code>loan_end</code> محاسبه میشود.</li>
            <li>این ویژگی نشان می‌دهد که دوره بازپرداخت وام چقدر طول کشیده است. مثلاً، وام‌های با مدت زمان طولانی‌تر ممکن است ریسک بیشتری داشته باشند یا نیاز به مدیریت دقیق‌تری داشته باشند.</li>
        </ul>
    </li>
    <li><h5>: <code>rate_times_loan_amount</code> (حاصل‌ضرب نرخ بهره در مبلغ وام)</h5>
        <ul>
            <li>این ستون ترکیبی از دو متغیر کلیدی <code>rate</code> و <code>loan_amount</code> است.</li>
            <li>می‌تواند به عنوان معیاری برای "هزینه کلی وام" استفاده شود، زیرا هم اندازه وام و هم نرخ بهره را در نظر می‌گیرد. برای مثال، یک وام با مبلغ بالا و نرخ بهره پایین ممکن است هزینه کلی مشابهی با یک وام با مبلغ پایین و نرخ بهره بالا داشته باشد.</li>
        </ul>
    </li>
    <li><h5>: <code>amount_per_day</code> (هزینه روزانه وام)</h5>
        <ul>
            <li>حاصل تقسیم <code>rate_times_loan_amount</code> بر <code>duration_days</code>.</li>
            <li>این ویژگی نشان می‌دهد که در هر روز، چه مقدار از وام به عنوان هزینه (نرخ بهره × مبلغ) به متقاضی تعلق می‌گیرد.</li>
            <li>می‌تواند به تحلیل وام توسط مشتری کمک کند، زیرا نشان می‌دهد که بار مالی روزانه برای مشتری چقدر است.</li>
        </ul>
    </li>
</ul>
</div>

<hr>  

<div dir="rtl">
<h4>📊 استانداردسازی ستون‌ها</h4>
<ul  dir="rtl">
    <li><h5> <code>loan_amount</code> (مبلغ وام)</h5>
            <ul>
                <li>مقادیر آن در بازه (559 تا 14,971) قرار دارند. استانداردسازی باعث میشود مقیاس آن را با سایر ویژگی‌ها هم‌خوانی پیدا کند</li>
            </ul>
    </li>
    <li><h5> <code>log_rate</code> (لگاریتم نرخ بهره)</h5>
            <ul>
                <li>در مرحله قبل عمل تبدیل لگاریتمی روی این ستون انجام شد و مقداری چولگی مثبت آن را کاهش دادیم ، اما هنوز نیاز به استانداردسازی برای همسان‌سازی با سایر ویژگی‌ها دارد.</li>
            </ul>
    </li>
    <li><h5> <code>rate_times_loan_amount</code> (حاصل‌ضرب نرخ بهره در مبلغ وام)</h5>
            <ul>
                <li>این ستون مقادیر بزرگی (به دلیل ضرب دو متغیر) دارد. استانداردسازی این ستون باعث میشود جلوی غلبه آن در مدل سازی گرفته شئد</li>
            </ul>
    </li>
    <li><h5><code>amount_per_day</code> (هزینه روزانه وام)</h5>
            <ul>
                <li>استانداردسازی آن اطمینان می‌دهد که وزن آن در مدل با سایر ویژگی‌ها متعادل باشد.</li>
            </ul>
    </li>
</ul>
</div>

<hr>  


<div dir="rtl">
<h4>🗑️ حذف ستون‌ها</h4>
<ul  dir="rtl">
    <li><h5> <code>rate</code> (نرخ بهره اصلی)</h5>
            <ul>
                <li>این ستون پس از تبدیل لگاریتمی به <code>log_rate</code> و استانداردسازی دیگر ضرورتی ندارد.</li>
                <li>حذف آن جلوی تکراری‌بودن (Redundancy) با ویژگی <code>log_rate</code> را می‌گیرد.</li>
            </ul>
    </li>
    <li><h5> <code>loan_id</code> (شناسه منحصر به فرد وام)</h5>
            <ul>
                <li>این ستون یک شناسه منحصر به فرد است و هیچ اطلاعات پیش‌بینی‌کننده‌ای برای وضعیت بازپرداخت (<code>repaid</code>) ندارد.</li>
            </ul>
    </li>
</ul>
</div>
<hr>  


<div dir="rtl">
<h4>💾 تقسیم و ذخیره‌سازی دیتافریم به مجموعه‌های آموزش و آزمون</h4>
<ul dir="rtl">
    <li><h5>📌 روش تقسیم:</h5>
        <ul>
            <li>استفاده از تابع <code>train_test_split</code> با نسبت <code>test_size=0.2</code> (20% داده‌ها برای مجموعه آزمون).</li>
            <li>کل داده‌ها (443 ردیف) به صورت زیر تقسیم شدند:
                <ul>
                    <li>مجموعه آموزش: 354 ردیف (80%)</li>
                    <li>مجموعه آزمون: 89 ردیف (20%)</li>
                </ul>
            </li>
            <li>پارامتر <code>random_state=42</code> برای تکرارپذیری تقسیم‌بندی استفاده شد.</li>
        </ul>
    </li>
    <li><h5>📌 ذخیره داده‌ها:</h5>
        <ul dir="rtl">
            <li>داده‌ها در پوشه <code>data</code> ذخیره شدند:
                <ul dir="rtl">
                    <li><code>train_set.csv</code> (مجموعه آموزش)</li>
                    <li><code>test_set.csv</code> (مجموعه آزمون)</li>
                </ul>
            </li>
            <li>پارامتر <code>index=False</code> اطمینان می‌دهد که ایندکس سطرها ذخیره نشوند.</li>
        </ul>
    </li>
</ul>
</div>


<hr>  
<div dir="rtl">
<h4>💾 ذخیره دیتافریم نهایی</h4>
<ul dir="rtl">
    <li><h5>📌 پارامترهای استفاده‌شده:</h5>
        <ul dir="rtl">
            <li><code>df_loans</code>: دیتافریم نهایی پس از تمام مراحل پیش‌پردازش (ایجاد ویژگی‌های جدید، استانداردسازی، حذف ستون‌ها).</li>
            <li><code>output_dir="data"</code>: داده‌ها در پوشه <code>data</code> ذخیره می‌شوند.</li>
            <li><code>filename="loans_final3"</code>: نام فایل خروجی <code>loans_final3.csv</code>.</li>
            <li><code>index=False</code>: ایندکس سطرها در فایل ذخیره نمی‌شوند.</li>
        </ul>
    </li>
    <li><h5>📌 روند ذخیره‌سازی:</h5>
        <ul dir="rtl">
            <li>اگر پوشه <code>data</code> وجود نداشته باشد، به صورت خودکار ایجاد می‌شود.</li>
            <li>دیتافریم با کدگذاری <code>utf-8-sig</code> ذخیره می‌شود تا از سازگاری با نرم‌افزارهای مختلف (مانند Excel) اطمینان حاصل شود.</li>
        </ul>
    </li>
</ul>
</div>

<hr>  
<br><br><br><br><br><br>
