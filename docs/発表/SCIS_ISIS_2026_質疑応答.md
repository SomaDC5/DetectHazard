# SCIS & ISIS 2026 質疑応答の手控え

- Extraction of Steep Slope Warning Areas Using Multi-Input Deep Learning（PID-128）
- 賀屋 颯真 関西大学大学院 総合情報学研究科
- 作成 2026-11-03 / **C-1 を実測で書き換え 2026-11-06**
- **原本はこの .md。** PDF/Word はここから生成する
  （`python docs/論文/build_roadmap.py --src docs/発表/SCIS_ISIS_2026_質疑応答.md --out docs/発表/SCIS_ISIS_2026_質疑応答.docx`）

| | |
|---|---|
| 第1部 | 自分の発表に来る想定質問と答え（23問・日英）|
| 第2部 | 直前の発表 PID-060 に回せる質問（12問・日英）|

**数字を言うときは、この論文の中の数字か、論文の後の数字かを必ず添えること。混ぜない。**

## 第1部 想定質問と答え

賀屋 颯真 "Extraction of Steep Slope Warning Areas Using Multi-Input Deep Learning"（PID-128）

### 使う前に ── 数字を2つの箱に分けておく

質問に答えるとき、その数字が論文の中のものか、論文の後のものかを必ず言うこと。混ぜると「論文に書いていない数字を言った」ことになる。

| 箱 | 中身 | 言い方 |
|---|---|---|
| **A：この論文** | 画素単位の 再現率・適合率・F値。広島 11,857タイル、8:2の無作為分割 | "In this paper …" |
| **B：この論文の後** | 箇所単位の評価、ゆらぎ幅、オーバーラップ、刈り取り、和集合、島根県 | "In work after this paper, on a rebuilt dataset, we measured …" |

箱Bを言うときの決まり文句（そのまま使える）日本語：「それはこの論文の後に測った結果になります。データセットを作り直していて、評価も箇所単位に変えているので、論文の数字とは直接つながりません。そのうえでお答えすると ──」English: "That is from work we did after this paper. We rebuilt the dataset and changed the evaluation to a per-slope basis, so the numbers are not directly comparable with the table in the paper. With that caveat ──"

## A. 直前の発表（PID-060 / Swin-Unet＋ Linear Overlap Fusion）から来そうな質問

相手は山口大学と防災科学技術研究所。同じ広島の航空写真を使い、災害後に実際に崩れた崩壊地を対象にしている（こちらは事前の法指定区域）。入力は RGBのみで、DEMの派生量（傾斜・曲率）を「今後の展望」に挙げている。

### A-1 なぜ Transformer（Swin-Unet）ではなく U-Netなのですか

#### 日本語

構造を変えることは一通り試しました。符号化器の種類、注意機構、解像度の扱いなど、構造6種・内部改良8種・損失の操作3点、合わせて17通りを同じデータ・同じ分割で測っています。結果は、同じ構成を乱数だけ変えて学習し直したときのゆらぎ幅の中に全部収まりました。 広島のゆらぎ幅は箇所F値で 0.0071です。パラメータ数が2.2倍のモデルでも差がつきませんでした。ですので「U-Netが最良」という主張ではなく、私たちのデータでは構造の選択が効く場所ではなかったということです。むしろ入力の種類と、出力をどう後処理するかのほうが桁違いに大きく動きました。Swin-Unetでも同じ結論になるかは測っていませんので、そこは分かりません。

#### English

We did try that direction. We compared seventeen variants in total — six architectures, eight internal modifications, and three changes to the loss — on the same data and the same split. Every one of them fell inside the band we get by retraining the identical configuration with a different random seed. That band is 0.0071 in per-slope F-value for Hiroshima. Even a model with 2.2 times the parameters showed no difference. So this is not a claim that U-Net is the best. It is that in our data, architecture was not where the gains were. What moved the numbers by an order of magnitude more was the choice of inputs and what we do to the output afterwards. We have not tested Swin-Unet specifically, so I cannot say it would behave the same way.

補足 ── 相手の論文も「アーキテクチャ上の優位を主張するものではない」と明記しており、報告された改善（F 0.40→0.70→0.72）はすべて学習方法・ラベル設計・融合から来ている。対立ではなく一致。必要なら "Your paper makes the same point in the introduction." と添えてよい。

### A-2 タイル端の精度低下は、重ねて推論すれば解決するのでは（＝私たちの Linear Overlap Fusion）

この研究で一番強く答えられる質問。 相手の提案そのものを、こちらは既に実装して測っている。

#### 日本語

まさにそれをやりました。この論文の後に、学習でもタイルをずらして重ね、推論でも重ねるようにしています。箇所単位のF値で、

- 学習側だけ重ねる ── 再現率 +0.0844

- 推論側だけ重ねる ── 再現率 +0.0747

- 両方 ── +0.1654単純な足し算（+0.1591）をわずかに超えていて、学習でずらしておくと推論のずらしがより効く方向に働きました。箇所F値では 0.6513 → 0.7071で、この時点で単体モデルとして最高です。ただし適合率は 0.6869 → 0.6436に下がりました。 予測面積は正解面積の 1.235倍で、塗り広げる方向に振れています。 ご発表の 再現率 +0.06 / 適合率 −0.02と、向きが完全に同じでした。

#### English

We did exactly that. After this paper we made the tiles overlap both in training and in inference. In per-slope F-value:

- overlap in training only — recall +0.0844

- overlap in inference only — recall +0.0747

- both — +0.1654 That is slightly more than the sum of the two, so shifting the tiles during training makes the shift at inference work better. Per-slope F went from 0.6513 to 0.7071, which was our best single model at that point. But precision fell from 0.6869 to 0.6436. The predicted area became 1.235 times the ground-truth area — it paints wider. That is the same direction as your result, recall up and precision down.

**ここから話を広げたいとき**

- 日本語：「ですので私たちは、広げたあとに捨てる段を足しました。」
- English: "So we added a second stage that throws candidates away again after widening them."

### A-3 重なったタイルの確率を重み付きで足し合わせれば、もっと良くなりませんか

相手の手法の前提を、こちらのデータでは測って否定している。 丁寧に言うこと。

#### 日本語

それが効くには、確率マップに中くらいの値が残っている必要があると思います。私たちの箱では、ここが成り立ちませんでした。出力の確率を数えると、0.02未満が 90.69%、0.98以上が 8.63%、その間は 0.68%しかありません。 二値化のしきい値を 50倍動かしても結果がほとんど変わらない状態でした。足し合わせても動く余地がなかったということです。ところが、確率に変換する前の値（ロジット）では、的中した塊と外れた塊が分かれていました。 中位値で 40.5対 27.7です。モデルは内部で区別できていて、最後の非線形がそれを潰していた。 そこで私たちは確率ではなくロジットのまま扱うことにしました。ご発表のデータでは足し合わせが効いているので、そちらの確率は飽和していないのだと思います。クラスの不均衡がこちらより緩いことが理由かもしれません。そこをお聞きしたいです。

#### English

For that to work, I think the probability map has to still contain intermediate values. In our case it did not. When we counted the output probabilities, 90.69% were below 0.02, 8.63% were above 0.98, and only 0.68% lay in between. Moving the binarisation threshold by a factor of fifty barely changed the output, so there was nothing left for averaging to move. However, before the sigmoid — in the logits — the correct blobs and the false ones were clearly separated: median 40.5 against 27.7. The model could tell them apart internally, and the final non-linearity was flattening that. So we stopped using probabilities and worked with the logits directly. In your data the fusion clearly works, so your probability map is probably not saturated. Perhaps your class imbalance is milder than ours. I would be interested to know.

### A-4 道路・河床・裸地の誤検出は、私たちのように補助クラスを立てれば減りませんか

#### 日本語

考え方は同じで、入れる場所が違います。私たちは出力のクラスではなく、入力の側で同じことをしています。航空写真そのものが土地被覆を伝えるので、「急だが下に守るものが無い斜面」を落とす役割を果たします。発表でお見せしたタイルでは、傾斜量図だけだと森林斜面に反応して適合率 0.39でしたが、写真を足すと 0.85になりました。

さらに論文の後では、建物の情報を別のモデルの入力として持たせ、その出力と和集合を取るという使い方も試しています。出力クラスを増やす形はまだ測っていないので、どちらが良いかは分かりません。やってみる価値はあると思います。

#### English

The idea is the same; we put it in a different place. We supply it as an input rather than as an output class. The aerial photograph itself carries the land cover, so it removes steep slopes with nothing underneath to protect. In the tile I showed, slope-angle alone responded to a forested slope and precision was 0.39; adding the photograph raised it to 0.85. After this paper we also tried giving building information to a separate model and taking the union of its output with ours. We have not tested the auxiliary-output-class form, so I cannot say which is better. I think it is worth trying.

### A-5 ほかの地域でも動きますか（＝私たちは広島の単一事例でしか評価していない）

#### 日本語

この論文の評価は無作為分割なので、同じ地域の中での精度と読んでいただく必要があります。それは論文にも明記しました。論文の後に、島根県でデータセットを作りました。学習には一度も使っていません。 広島で学習したモデルをそのまま島根に当てると、箇所F値は 0.4820まで落ちます。地域が変わると確かに落ちます。ただしひとつ良い結果がありました。広島で選んだ後処理のしきい値の絶対値が、島根にもそのまま有効でした。選び直していません。そして一連の後処理を通した結果、島根は 0.4820 → 0.6564まで上がりました。

#### English

The evaluation in this paper uses a random split, so it should be read as accuracy within the same region. We state that in the paper. After this paper we built a dataset for Shimane Prefecture, which was never used in training. Applying the Hiroshima-trained model directly to Shimane, the per-slope F-value drops to 0.4820. Performance does fall when the region changes.

But there was one good result. The absolute threshold we chose on Hiroshima transferred to Shimane without being re-selected. With the full post-processing chain, Shimane went from 0.4820 to 0.6564.

### A-6 対象が違いますね。崩壊地と警戒区域の違いは何ですか（座長が振ってくる可能性が高い）

#### 日本語

はい、同じ広島でも答えが別物です。ご発表は実際に崩れた跡で、写真に写っています。私たちは行政が事前に指定した区域で、写真に写っているとは限りません。 指定には2つの条件があり、斜面が十分に急であること、そしてその下に守るべき家屋等があることです。2つ目は地形には写りません。だから航空写真を足しています。時間軸も逆です。ご発表は災害後の把握、私たちは災害前の指定です。ですので同じデータセットで比較することはできません。

#### English

Yes — even though both are Hiroshima, the ground truth is a different thing. Yours is the scar of an actual collapse, which is visible in the photograph. Ours is a zone the government designates in advance, which is not necessarily visible at all. A designation requires two conditions: the slope must be steep enough, and there must be houses or assets below that would be damaged. The second one is invisible in terrain data, which is why we add the photograph. The time axis is also opposite: yours is post-disaster mapping, ours is pre-disaster designation. So the two cannot be compared on one dataset.

## B. 手法とモデルへの質問

### B-1 傾斜量図（SAM）は DEMから計算しただけで、新しい情報はないはずです。なぜ精度が上がるのですか

#### 日本語

おっしゃるとおり、新しい観測は入っていません。形が変わっただけです。それでも F値は 0.5709 → 0.6016、再現率は 0.5918 → 0.6744と上がりました。同じ事実を、モデルが学びやすい形に書き直したということだと考えています。急傾斜地の判定に効くのは局所の高低差であって、標高そのものではないので、差分を先に取ってから渡したほうが有利になります。ひとつ注意点があります。「DEM」の結果は正規化の仕方に強く依存します。 私たちはタイルごとの最低標高を引いてから割っています。生の標高をそのまま渡すと、タイル間の標高差が分散を支配してタイル内の起伏が埋もれます。 実測すると、タイル間とタイル内のばらつきの比は、生の標高で4.45、引き算した後で 0.91でした。効いているのは引き算のほうで、定数で割ること自体は性能に影響しません。

#### English

You are right — there is no new measurement. Only the form changes. Even so, F went from 0.5709 to 0.6016 and recall from 0.5918 to 0.6744. We read this as rewriting the same facts in a form the model learns from more easily. What matters for a steep slope is the local height difference, not the absolute height, so taking the difference first helps. One caution: the "DEM" result depends strongly on how you normalise it. We subtract the minimum elevation of each tile before scaling. If you feed raw elevation, the between-tile elevation offset dominates the variance and the within-tile relief is buried. We measured it: the ratio of between-tile to within-tile standard deviation is 4.45 for raw elevation and 0.91 after the subtraction. It is the subtraction that matters; dividing by a constant does not change performance at all.

### B-2 符号化器を分けずに、最初からチャネルとして重ねてはどうですか（early fusion）

#### 日本語

それも測りました。結果はゆらぎ幅の中で、分けても分けなくても差が出ませんでした。設計の根拠としては、傾斜量図は1チャネルで値域が角度、航空写真は3チャネルで値域が輝度なので、浅い層で混ぜないほうが筋が通ると考えて分けています。ただしその判断を支持する差は測れていません。正直に言えば、ここは効く場所ではありませんでした。

#### English

We measured that as well. The result was inside the fluctuation band — separating the encoders or not made no measurable difference. Our design reason is that the slope map is one channel of angles and the photograph is three channels of brightness, so not mixing them in the shallow layers seemed principled. But we could not measure a difference that supports that choice. Honestly, this was not a place where it mattered.

### B-3 Focal Tversky Lossの α = 0.7はどう決めましたか。調整しましたか

#### 日本語

原著の設定に倣って、見逃しの重みを誤検出より大きくしています。意図は非対称なコストをそのまま損失に入れることです。ただし論文の後に測って分かったのは、損失の操作は3点試してすべてゆらぎ幅の中だったということです。再現率を動かしたのは損失ではなく、二値化のしきい値と、タイルの重ね方でした。 ですので今は、損失で非対称性を表現するより、しきい値を再現率寄りに選び直すほうを主に使っています。

#### English

We followed the setting in the original paper, weighting misses more heavily than false alarms. The intent is to put the asymmetric cost directly into the loss. However, what we learned after this paper is that we tried three variations of the loss and all of them fell inside the fluctuation band. What actually moved recall was the binarisation threshold and the tile overlap, not the loss. So now we mainly express the asymmetry by re- selecting a recall-oriented threshold rather than through the loss.

### B-4 単に二値化のしきい値を下げれば、再現率は上がるのではないですか

#### 日本語

それを最初に試して、動かなかったことがこの研究の転換点になりました。出力の確率は 0.02未満が90.69%、0.98以上が 8.63%、その間が 0.68%しかありません。 しきい値を 50倍動かしても結果がほとんど変わりませんでした。分布が両端に潰れていて、しきい値を動かす余地が無かったのです。

その代わり、確率に変換する前の値では的中と外れが分かれていました（中位値 40.5対 27.7）。そこで私たちは、確率ではなくロジットのまま後処理する方向に切り替えました。

#### English

That was the first thing we tried, and the fact that it did nothing became the turning point of this research. 90.69% of the output probabilities are below 0.02, 8.63% are above 0.98, and only 0.68% lie in between. Moving the threshold by a factor of fifty barely changed the result — the distribution was collapsed to both ends, so there was no room for the threshold to act. But before the sigmoid, the correct and the false blobs were separated — median 40.5 against 27.7. So we switched to post-processing on the logits instead of the probabilities.

### B-5 精度を上げるために、次は何をしますか

#### 日本語

3段階で進めています。いずれもこの論文の後の結果です。1. タイルを重ねて学習・推論する ── 箇所の再現率 +0.1654、箇所F 0.6513 → 0.7071 2. ロジットで下位の塊を刈り取る ── 0.7071 → 0.7472。しきい値を別のタイルで選ぶ公正な評価でも +0.0374で、学習は1回も増やしていません3. 入力の異なるモデルの論理和と、モデル間の合意数 ── 広島 0.7908、島根 0.4820 → 0.6564捨てるだけの後処理が届く上限は F = 2R/(1+R)で決まるので、前段の再現率を上げないと上限ごと動きません。 2段目だけでは 0.8793が限界で、和集合で前段を広げると上限が 0.9450まで上がります。いまはその差を詰めているところです。

#### English

We are working in three steps. All of these are after this paper. 1. Overlapping tiles in training and inference — per-slope recall +0.1654, per-slope F 0.6513 → 0.7071 2. Pruning low-scoring blobs by their logits — 0.7071 → 0.7472. Even under a fair protocol where the threshold is chosen on separate tiles it is +0.0374, and we did not train a single additional model

3. Union of models with different inputs, plus an agreement-count channel — Hiroshima 0.7908, Shimane 0.4820 → 0.6564 A post-processing step that only discards is bounded by F = 2R/(1+R), so the ceiling moves only if the recall of the first stage moves. With the second stage alone the ceiling is 0.8793; widening the first stage with the union raises it to 0.9450. Closing that gap is what we are doing now.

## C. データと評価の作法への質問（一番厳しいところ）

### C-1 警戒区域を含まないタイルを 51.3%捨てていますね。それは有利な評価ではありませんか

正面から認めること。ごまかさない。

#### 日本語

ご指摘のとおりで、論文にも限界として書きました。整理すると、

- 再現率は影響を受けません。 捨てたタイルには正解画素が1つも無いので、TPも FNも変わりません。表の再現率は全域に適用してもそのまま成り立ちます。

- 適合率は影響を受けます。 誤検出をどこで数えるかが変わるからです。表の適合率は「警戒区域の周辺で測った値」であって、全域ではもっと低くなる可能性があります。数字も出してあります。テスト集合の中での画素単位の誤検出率は DEM 6.3%、SAM 7.5%、DEM+APM 4.5%、SAM+APM 5.5%です。また、テスト画素のうち警戒区域は 11.6%で、残り88.4%の上で誤検出を数えています。ただし「警戒区域が1つも無い無人の急斜面」は入っていません。 そこは地形だけの入力ほど誤検出が出るはずです。

**そして全域での誤検出は、この論文の後に測りました（2026-11-06）。**
警戒区域を1画素も含まない背景タイル 23,332枚に推論をかけると、
かたまりが **13,994件・83.5 km²** 出ます。評価に使ったタイルの予測は
9,669件・90.6 km² なので、**評価の外側にほぼ同じ量の「誤検出」があります。**
ロジットで下位30%を刈り取っても 56%（7,837件・74.1 km²）が残ります。

ただし**背景タイルは広島の全域ぶんで、評価に使ったテストは 2,964枚なので、
足して適合率を出すことはできません。** 1枚あたりで比べると、背景タイルは
**0.60件**、警戒区域を含むタイルの中の外れは **1.16件**（3,445 / 2,964）で、
**むしろ背景タイルのほうが少ない**です。枚数が7.9倍あるので量が同じになります。

**ただし中身を見ると、単純な誤りとは言いにくいものでした。**
ロジットの上位500件を、実際の指定区域の中で的中したかたまりと並べると、

- 傾斜の90%点は **38.2度 対 33.3度**。30度以上の割合は **94.4% 対 76.5%** で、**上位群のほうが急**です
- 50m以内に5戸以上あるものの割合は **3.4% 対 3.0%** で**同水準**です（建物ラスタの連結成分で数えているので、指定区域自身でも 3.00% しか満たしません。絶対値では読めず、指定区域との比でだけ読んでいます）
- モデルの確信（ロジットの中位値）は **52.6 対 40.5** で、**上位群のほうが高い**です
- 8近傍に指定区域があるものは全体で 33.7% ですが、上位500では **11.8%** です。**境界のはみ出しでもありません。** 学習でタイルの外側を負例として見ている分の偏りを外し、**学習タイルに接していない 9,841件（70.3%）だけで取り直しても同じ**です（傾斜の90%点 37.94度・30度以上 94.6%・ロジット中位 52.19）。上位500の 88.2% は指定区域にも学習タイルにも接していません

**正解が部分ラベルである可能性が高いと考えています。**
ただし急で人家があっても指定されない行政上の理由はいくつもあります
（既に対策済み・別制度の区域・土地の所有・申請の有無）。
ですので言えるのは「**指定の漏れ**」ではなく
「**指定基準の地形条件を満たすが指定されていない斜面**」までです。
裏を取るには現地か、土砂災害警戒区域などの別制度の区域データが要ります。
それは持っていません。

#### English

That is a fair point and we list it as a limitation in the paper. To be precise:

- Recall is not affected. The removed tiles contain no ground-truth positive pixel, so neither TP nor FN changes. The recall values in the table still hold if the method is applied to a whole region.

- Precision is affected, because it changes where false positives are counted. The precision in the table is the value measured around warning areas, and it may be lower over a whole region.

We do give numbers. Inside the test set the pixel-wise false-positive rate is 6.3% for DEM, 7.5% for SAM, 4.5% for DEM+APM and 5.5% for SAM+APM. Warning-area pixels are 11.6% of the test pixels, and false alarms are counted over the other 88.4%. But uninhabited steep forest with no designation at all is not represented. Terrain-only inputs would fire there.

**We measured that after this paper (6 November 2026).** Running inference on
the 23,332 background tiles — tiles that contain no designated pixel at all —
produces **13,994 blobs covering 83.5 km²**. The predictions inside the tiles we
evaluate on are 9,669 blobs over 90.6 km², so **there is almost the same amount
of "false alarm" outside the evaluated area**, and 56% of it survives our logit
pruning.

The two sets have different denominators — the background tiles cover the whole
of Hiroshima while we evaluate on 2,964 tiles — so **we cannot simply add them up
and quote a precision.** Per tile the background tiles give **0.60 blobs**
against **1.16** false ones inside tiles that do contain a designated zone
(3,445 / 2,964), so the *density* of false alarms is actually lower outside.
There are 7.9 times as many background tiles, which is why the totals match.

**But the content is hard to call an error.** Comparing the top 500 by logit
with the true positives inside designated zones:

- 90th-percentile slope **38.2° against 33.3°**, and **94.4% against 76.5%** above 30° — the background group is *steeper*
- **3.4% against 3.0%** have five or more buildings within 50 m — the same level (we count connected components in a building raster, and even real designated zones only reach 3.00%, so we read this as a ratio, never as an absolute)
- median logit **52.6 against 40.5** — the model is *more* confident about them
- 33.7% of all background detections touch a designated zone, but only **11.8%** of the top 500 — so they are not spill-over across a boundary

**We think the ground truth is most likely a partial label.** But there are
administrative reasons a steep inhabited slope may not be designated (already
mitigated, covered by another scheme, land ownership, whether an application was
made). So the most we can claim is "slopes that meet the terrain condition of
the designation criteria but are not designated" — not "missed designations".
Confirming it needs a field survey or zone data from another scheme, which we
do not have.

### C-2 正解マスクにガウシアンをかけて約3画素太らせていますね。再現率が水増しされませんか

#### 日本語

境界の曖昧さを扱うためで、4条件すべてに同じ処理をかけていますので、条件間の比較は影響を受けません。 絶対値は動きます。平滑化の前は警戒区域がテスト画素の 7.8%、後は 11.6%です。この論文の後、評価そのものを箇所単位に変えました。連結成分に分け、被覆率で正解と予測を対応づけます。 この測り方なら、境界が3画素太いかどうかはほとんど効きません。 ご指摘の懸念は、箇所単位に移したことで構造的に外れています。

#### English

It is there to handle boundary ambiguity, and the same operation is applied to all four conditions, so the comparison between conditions is unaffected. The absolute values do move: warning areas are 7.8% of the test pixels before the smoothing and 11.6% after. After this paper we changed the evaluation itself to a per-slope basis: we split the map into connected components and match prediction to ground truth by coverage ratio. Under that measure, whether the boundary is three pixels wider hardly matters. Moving to per-slope evaluation removes this concern structurally.

### C-3 無作為分割だと、隣り合うタイルが学習とテストに分かれますね

#### 日本語

はい。空間的な自己相関があるので、報告した値は未知の地域への汎化ではなく、同じ地域の中での精度です。論文にそのまま書きました。

論文の後、島根県という別の県で評価しています。学習には一度も使っていません。そこでは箇所F値が 0.4820で、広島の 0.6506から 23帯ぶん落ちます。 ですので、地域をまたぐと確かに落ちるというのが正直な答えです。いまは島根を主系統として扱い、そちらを上げることを目標にしています。

#### English

Yes. Because of spatial autocorrelation, the reported values are accuracy within the same region, not generalisation to unseen regions. We say exactly that in the paper. After this paper we evaluated on Shimane Prefecture, which was never used in training. There the per-slope F-value is 0.4820, about 23 fluctuation-bands below the 0.6506 we get in Hiroshima. So the honest answer is that performance does drop across regions. We now treat Shimane as our primary measure and aim to raise it.

### C-4 表の差は意味のある差ですか。1回しか学習していないのでは

痛いところ。論文では乱数を揃えただけ。正直に言ってから、後の仕事に逃がす。

#### 日本語

この論文では、4条件で乱数の種を揃えて1回ずつ学習しています。 ですので「入力以外は同一」という比較にはなっていますが、学習のばらつきを測ってはいません。 そこは弱点です。論文の後に測りました。同じ構成を種だけ変えて学習し直すと、広島の箇所F値は 0.0071、島根は0.0198のばらつきが出ます。 この幅を基準にすると、表の差は ── 傾斜量図と標高の差 0.0307、写真を足した差 0.09〜0.112 ── いずれも幅を大きく超えていますので、入力の差は本物だと考えています。構造をいじった17通りが全部この幅の中に収まったのとは対照的です。

#### English

In this paper we trained each of the four conditions once, with the random seed fixed across them. That makes it a clean "everything but the input is identical" comparison, but it does not measure the run-to-run variation. That is a weakness. We measured it after this paper. Retraining the same configuration with a different seed gives a spread of 0.0071 in per-slope F for Hiroshima and 0.0198 for Shimane. Against that band, the differences in the table — 0.0307 between slope map and elevation, and 0.09 to 0.112 for adding

the photograph — are all well outside it, so we believe the input differences are real. That contrasts with the seventeen architectural variants, which all fell inside the band.

### C-5 画素のF値 0.69は低いのではありませんか

#### 日本語

低いです。実運用にはまだ足りません。 ただ、画素のF値はこの用途で見るべき数字ではないと今は考えています。技術者が確認するのは「どの斜面を見に行くか」であって、「何画素塗れたか」ではありません。そこで論文の後に、箇所単位の評価に切り替えました。連結成分に分けて被覆率で対応づけ、「その斜面を見つけたか」を数えます。この軸で測ると、一連の後処理を通して広島は 0.7908です。それでもまだ足りませんが、少なくとも現場の問いと同じ軸で測れるようになりました。

#### English

It is low. It is not yet enough for operational use. But I now think pixel F-value is not the right number for this application. What an engineer checks is *which slopes to go and inspect*, not how many pixels were coloured. So after this paper we moved to per-slope evaluation: split into connected components, match by coverage ratio, and count whether that slope was found at all. On that axis, with the full post- processing chain, Hiroshima is at 0.7908. It is still not enough, but at least we are now measuring on the same axis as the question the field actually asks.

## D. 用途と意義への質問

### D-1 再現率はどこまで上げれば実用になりますか

#### 日本語

100%に近づけるべきだと考えています。2つの誤りのコストが対称ではないからです。誤検出は技術者の確認で落とせるので点検の手間が増えるだけですが、見逃しは確認の段階にすら届かないので、

その斜面は指定されないまま残ります。基礎調査はおよそ5年ごとなので、見逃しは最長で5年残ります。適合率は目標ではなく予算の制約です。適合率が pなら、技術者は実際に危険な面積のおよそ 1/p倍を見ることになります。ですので「適合率がいくつ必要か」は、その自治体に点検の余力がどれだけあるかで決まり、こちらが決める数字ではありません。

#### English

We think it should approach 100%, because the two errors do not cost the same. A false positive is removed during verification and only adds inspection work; a miss never reaches the verification stage at all, so that slope stays undesignated. The basic survey is repeated about every five years, so a miss can persist for up to five years. Precision is not a target but a budget constraint. With precision p, engineers inspect roughly 1/p times the area that is truly hazardous. So "how much precision is needed" is decided by how much inspection capacity the prefecture has, not by us.

### D-2 なぜモデルの出力をそのまま使わないのですか。人の確認は残すのですか

#### 日本語

残します。この手法は指定の手続きを置き換えるものではなく、机上調査の段階を支えるものです。出力は候補であって、技術者が既存の現地調査で確認します。狙いは2つで、広域で航空写真を見比べる手間を減らすことと、人の目視で見落とされうる斜面を出すことです。ですので、誤検出が多少出ても安全性は損なわれません。 点検の手間が増えるだけです。一方で見逃しは直接、危険の見落としになります。この非対称性が、再現率を主指標にしている理由です。

#### English

We keep it. The method is not a replacement for the designation procedure; it supports the desk-survey stage. The output is a set of candidates that engineers verify in the field survey that already exists. The aims are to reduce the manual effort of comparing aerial photographs over wide areas, and to surface slopes that manual screening might overlook.

So a certain number of false positives does not compromise safety — it only adds inspection work. A miss, on the other hand, is directly a missed hazard. That asymmetry is why recall is our primary metric.

### D-3 なぜその斜面が選ばれたのか、説明できますか

#### 日本語

今はできません。説明可能性は今後の課題として挙げています。ただ、部分的には入力条件の比較そのものが説明になっています。 傾斜量図だけのときに森林斜面へ反応し、航空写真を足すとそれが消える ── これはモデルが「下に守るものがあるか」を見ていることの証拠です。全テスト集合でも、写真を足すと誤検出画素が DEMで 28.3%、SAMで 26.1%減り、同時に正検出は増えました。ただし一様ではなく、誤検出が減ったのはタイルの 57%／61%です。残りでは増えています。

#### English

Not yet. Explainability is listed as future work. But the input comparison itself is partly an explanation. Terrain-only fires on a forested slope, and adding the photograph removes it — that is evidence that the model is looking at whether there is something below to protect. Over the whole test set, adding the photograph removed 28.3% of false-positive pixels for DEM and 26.1% for SAM while true positives went up at the same time. But it is not uniform: false positives decreased in 57% and 61% of the tiles respectively, and increased in the rest.

### D-4 植生に覆われた斜面はどうしますか

#### 日本語

いまの限界そのものです。密な林冠の下では航空写真が地表の状態を伝えられないので、条件（ii）に答えられません。その例では SAM+APMでも再現率 0.49でした。この失敗は4条件すべてに出ます。ネットワークの限界ではなくデータの限界です。

対応としては、地質情報と土地利用のデータを足すことを考えています。航空写真が見通せない場所でも、地質と土地利用は別経路の証拠になります。曲率や陰影起伏図といった地形表現の追加も予定しています。

#### English

That is exactly our current limit. Under a dense canopy the aerial photograph cannot convey the state of the ground, so condition (ii) cannot be answered. In that example recall was 0.49 even with SAM+APM. This failure appears in all four conditions — it is a limit of the data, not of the network. Our plan is to add geological and land-use data, which provide evidence through a different channel where the photograph cannot see. We also plan to add terrain representations such as curvature and hillshade.

## E. 答えを持っていない質問 ── 正直に言う

推測で答えない。 「測っていません」と言ってから、代わりに測ったことを言う。

**聞かれそうなこと**

| 聞かれそうなこと | 日本語 | English |
|---|---|---|
| 処理時間・GPUメモリ | 「測っていません。 学習は1,000エポック、バッチ32です。推論時間は記録していません」 | "We did not measure it. Training was 1,000 epochs with batch size 32. We did not record inference time." |
| Swin-Unetや YOLOとの比較 | 「同一データでは比較していません。構造17通りはすべて U-Net系の範囲内です」 | "We have not compared them on the same data. Our seventeen variants are all within the U-Net family." |
| 他の災害種別（地すべり・土石流） | 「急傾斜地崩壊だけです。件数が最も多い類型なので選びました（年平均996件）」 | "Steep-slope failure only. We chose it because it is the most frequent type — about 996 cases a year." |
| 指定区域そのものの妥当性 | 「私たちは指定を正解として扱っています。 指定が物理的な危険度と一致するかは、私たちが判断できる問題ではありません」 | "We treat the designation as ground truth. Whether the designation matches physical hazard is not something we are in a position to judge." |
| 5mメッシュより細かいデータ | 「使っていません。 入力の高解像度化は別に測っていて、箇所の再現率で+0.0223でした。タイルを重ねる効果（+0.0844）の約4分の1です」 | "We did not use any. We measured higher input resolution separately: +0.0223 in per-slope recall, about a quarter of what tile overlap gave us (+0.0844)." |

## 付録：言ってよい数字の一覧

### 箱A ── この論文の中（画素単位・広島・無作為分割・11,857タイル）

| 入力 | 再現率 | 適合率 | F値 | 中央のみのF値 |
|---|---:|---:|---:|---:|
| DEM | 0.5918 | 0.5514 | 0.5709 | 0.5894 |
| SAM | 0.6744 | 0.5430 | 0.6016 | 0.6212 |
| DEM+APM | 0.6968 | 0.6687 | 0.6825 | 0.7010 |
| SAM+APM | 0.7500 | 0.6413 | 0.6914 | 0.7084 |

- 画素単位の誤検出率：DEM 6.3% / SAM 7.5% / DEM+APM 4.5% / SAM+APM 5.5%
- 写真を足して消えた誤検出画素：DEM −28.3%、SAM −26.1%（ただしタイルの 57% / 61%でのみ減少）
- 誤検出が支配的なタイル（適合率 0.35未満）：13.4%→6.9%（DEM）、16.6%→9.6%（SAM）
- 林冠下の例：SAM+APMでも再現率 0.49
- テスト画素に占める警戒区域：11.6%（平滑化前 7.8%）／捨てたタイル 51.3%

### 箱B ── この論文の後（箇所単位・データセット作り直し）

| | 広島 | 島根（学習に未使用）|
|---|---|---|
| ゆらぎ幅（帯）| 0.0071 | 0.0198 |
| 出発点 箇所F | 0.6513 | 0.4820 |
| ＋タイル重ね（学習・推論）| 0.7071（箇所R +0.1654）| 箇所R +0.0741、箇所F 横ばい |
| ＋ロジットで刈り取り | 0.7472（公正評価 +0.0374、学習ゼロ）| 広島のしきい値がそのまま有効 |
| ＋和集合6本＋合意数 | 0.7908 | 0.6564 |
| 刈り取りだけの上限 | 0.8793 | — |
| 和集合後の上限 | 0.9450 | 0.8620 |
| **背景タイルの誤検出（2026-11-06 実測）** | **13,994件・83.5 km²**（刈り取り後も 56%が残る）| 未測定 |
| **上位20%だけ見たときの箇所P（2026-11-06 実測）** | **0.9167**（1,944件・48.9 km²。全件は 0.6437）| **0.7651**（6,862件。全件は 0.5413）|

- 構造6種・内部改良8種・損失3点＝17通りすべてゆらぎ幅の中

- 確率の飽和：0.02未満 90.69%／ 0.98以上 8.63%／ その間 0.68%

- ロジットの中位値：的中 40.5対 外れ 27.7

- 標高の正規化：タイル間/タイル内のばらつきの比は、生の標高 4.45、タイル最低標高を引いた後 0.91

### 最後に ── 答え方の型

1. **質問を1文で言い直す**（通訳と聴衆のため）── "If I understand correctly, you are asking whether …"
2. **測ったか測っていないかを先に言う** ── 測っていないなら "We did not measure that." で始める。推測から始めない
3. **数字を言うときは、どの箱かを必ず添える** ── "In this paper …" / "In work after this paper …"
4. **限界は自分から言う**（相手に言われる前に言うほうが強い）
5. **最後に相手へ返す** ── "I would be interested to know how that behaves in your setting."

## 第2部 こちらから聞く質問

- 相手：Xiaosong Liu ほか（山口大学）、若月 強（防災科学技術研究所）
- 題：Multi-Scale Swin-Unet with Linear Overlap Fusion for Landslide Segmentation from Aerial Imagery

**使い方**

- 質疑は1人1問が普通。下から1つだけ選んで、残りは懇親会や休憩で使う
- ★ の数は踏み込みの深さ。★☆☆ は誰に聞かれても困らない質問、★★★ は論文の弱いところを突く質問
- 自分の研究の宣伝にしない。自分の数字は、相手が「そちらはどうですか」と返してきたときだけ出す
- 英語は短く切ってある。長い一文にしないこと

**前置き（共通）**

- 日本語：「関西大学の賀屋です。興味深いご発表をありがとうございました。1点うかがいます。」
- English: "Thank you for the interesting talk. Soma Kaya, Kansai University. I have one question."

### 第1群：まず聞きたい3つ

### Q1 ★★☆ 学習をやり直したときのばらつきと比べて、+0.02はどのくらいですか

なぜ効くか ── 論文の数字はすべて1回の実行の単一値で、乱数を変えて学習し直したときの幅が報告されていない。Linear Overlap Fusionの積み増しは F値 +0.02。この幅が分からないと、+0.02が本物かどうか判断できない。論文が自分では答えていない、最も素直な質問。

#### 日本語

表IIで、Linear Overlap Fusionによる改善は F値で +0.02と理解しました。同じ条件を乱数の種だけ変えて学習し直したとき、F値はどのくらいばらつきますか。 その幅と比べて +0.02はどの程度の大きさでしょうか。

#### English

In Table II, the improvement from Linear Overlap Fusion is +0.02 in F1. If you retrain the same configuration with a different random seed, how much does the F1-score move? I would like to know how +0.02 compares with that spread.

**返ってきそうな答えと、その先**

| 相手の答え | こちらの返し |
|---|---|
| 「測っていません」 | 「ありがとうございます。私たちは同じ構成を種だけ変えて測っていて、幅が 0.0071でした。もしよろしければ後ほど測り方をお話しさせてください」 |
| 「ばらつきは小さいです」 | 「何回くらい回されましたか」 |

**注意** ── 融合は学習をやり直さない後処理なので、同じ重みに対しては決定的。聞いているのは「融合の揺れ」ではなく**「学習そのものの揺れ」**である。そこを明確に言うこと。

- 日本語で補うなら「融合自体は決定的だと思いますので、学習側のばらつきのことです」
- English: "I mean the variation in training itself, since the fusion step is deterministic for a given model."

### Q2 ★★☆ 確率マップは飽和していませんでしたか

なぜ効くか ── Linear Overlap Fusionは「弱いが一貫した反応が積み上がる」ことで成立する。それは確率マップに中くらいの値が残っている場合にだけ起きる。 賀屋さんの箱では 0.02未満が90.69%、0.98以上が8.63%、間が0.68%しかなく、足し合わせても動かなかった。相手の手法が効いたこと自体が、相手のデータの性質を示している。 攻撃ではなく、現象の違いを聞く質問。

#### 日本語

Linear Overlap Fusionは、隣接タイルの弱い反応が積み上がることで再現率が上がる、というご説明だったと思います。そうすると、確率マップに中間的な値がある程度残っていることが前提になりま

すね。 融合の前の確率の分布はどのようになっていましたか。0と 1の両端に寄っていましたか、それとも中間にも分布していましたか。

#### English

If I understood correctly, Linear Overlap Fusion raises recall because weak but consistent responses from neighbouring tiles accumulate. That would require the probability map to still contain intermediate values. What does the distribution of probabilities look like before fusion — is it concentrated near 0 and 1, or is there mass in between?

**返ってきそうな答えと、その先**

| 相手の答え | こちらの返し |
|---|---|
| 「中間にも分布しています」 | 「ありがとうございます。私たちのデータでは両端に潰れていて、しきい値を50倍動かしても出力がほとんど変わりませんでした。 クラスの不均衡の差かもしれません。正解画素の割合はどのくらいでしょうか」 |
| 「見ていません」 | 「もし飽和していたら足し合わせの効きが変わるはずなので、見てみられると面白いかもしれません」 |

### Q3 ★★☆ 同じ学習とタイル融合を、ふつうの U-Netでも試されましたか

なぜ効くか ── 論文自身が「アーキテクチャの優位は主張しない」と明記している。すると改善はすべて構造の外側（マルチスケール学習・補助クラス・融合）から来ていることになる。ならば Swin-Unetでなくても同じ効果が出るのではという問いは自然で、相手にとっても次の実験の示唆になる。こちらの「構造17通りが全部ゆらぎ幅の中だった」という結果と、まっすぐつながる。

#### 日本語

序論で「アーキテクチャ上の優位を主張するものではない」と明記されている点が印象的でした。そうしますと、報告された改善 ── マルチスケール学習、補助クラス、Linear Overlap Fusion ── はいずれも構造に依存しない要素だと理解しました。同じ学習方法と同じ融合を、たとえばふつうの U- Netに適用されたことはありますか。 あるとすれば、差はどのくらいでしたか。

#### English

I was struck by the statement in the introduction that you do not claim architectural superiority. If I follow that, the gains you report — multi-scale training, the auxiliary class, and Linear Overlap Fusion — are all components that do not depend on the architecture. Have you applied the same training recipe and the same fusion to a plain U-Net? If so, how large was the difference?

**返ってきそうな答えと、その先**

| 相手の答え | こちらの返し |
|---|---|
| 「やっていません」 | 「ありがとうございます。私たちは構造を17通り変えて測ったのですが、すべて学習のゆらぎ幅の中に収まりました。 もし同じ傾向でしたら、ご発表の結論はさらに強くなると思います」 |
| 「やって、Swinのほうが良かった」 | 「何点くらい差がありましたか。 そしてそれは学習のばらつきより大きかったでしょうか」 |

### 第2群：相手の「今後の展望」に踏み込む

### Q4 ★☆☆ 傾斜量図は、写真と同じ符号化器に入れますか、別系統にしますか

なぜ効くか ── 相手は結論で「DEMの派生量（傾斜・曲率・相対標高）」を展望に挙げている。これは賀屋さんの出発点そのもの。 相手にとっても有益な情報を渡せる、最も友好的な質問。

#### 日本語

今後の展望で、傾斜や曲率といった DEMの派生量を加えるとお書きでした。入れるとしたら、航空写真と同じ符号化器にチャネルとして重ねるお考えですか、それとも別の符号化器に通して後で結合するお考えですか。

#### English

In your future work you mention adding DEM derivatives such as slope and curvature. If you do, would you stack them as extra channels into the same encoder as the photograph, or pass them through a separate encoder and join the features later?

返しに使える情報（聞かれたら）日本語：「私たちは値域も意味も違うので別系統にしています。ただその選択を支持する差は測れていませんで、早期結合との差はゆらぎ幅の中でした。それより標高の正規化のほうがずっと効きました。 タイルごとの最低標高を引かないと、タイル間の標高差が分散を支配して、タイル内の起伏が埋もれます。実測で、タイル間/タイル内のばらつきの比は生の標高で 4.45、引いた後で 0.91でした」English: "We use separate encoders because the value ranges and the meanings differ. But we could not measure a difference that supports that choice — early fusion was inside our noise band. What mattered far more was how the elevation is normalised. If you do not subtract the per-tile minimum, the between-tile elevation offset dominates the variance and the within-tile relief is buried. We measured the ratio of between-tile to within-tile standard deviation: 4.45 for raw elevation, 0.91 after the subtraction."

### Q5 ★★☆ 崩壊痕と「裸地」は、ラベルの上でどう区別されたのですか

なぜ効くか ── 補助クラス「主要地物」は道路・河床・裸地をまとめている。ところが新しい崩壊痕も、まさに裸地である。 判読者がどう線を引いたのかは、この論文の設計の核心にある曖昧さ。論文は触れていない。

#### 日本語

補助クラスの「主要地物」には裸地が含まれるとのことでした。一方で、新しい崩壊痕もまた裸の地表です。判読の段階で、この2つはどのような基準で区別されたのでしょうか。形でしょうか、位置でしょうか、それとも災害前の画像との比較でしょうか。

#### English

Your auxiliary class, "major ground objects", includes bare land. But a fresh landslide scar is also bare ground. At the interpretation stage, on what basis were those two distinguished — the shape, the location, or a comparison with pre-disaster imagery?

### Q6 ★★☆ 継ぎ目そのものでの改善は切り分けられますか

なぜ効くか ── 論文が自分で書いている ── 「図にタイル境界を重ねていないし、境界距離の指標も計算していない。継ぎ目そのものでの改善を切り分けたものではない。」 自分で認めた限界を、どう埋める予定かを聞く形にすれば、失礼にならない。

#### 日本語

Linear Overlap Fusionの効果について、境界距離の指標は今後の課題と書かれていました。 もし切り分けるとしたら、どのような測り方をお考えですか。たとえば継ぎ目から一定距離の画素だけで精度を出すといった形でしょうか。

#### English

You note that a boundary-distance metric is left for future work. If you were to isolate the effect at the seams, how would you measure it? For example, would you compute accuracy only over pixels within a fixed distance of a tile boundary?

### Q7 ★☆☆ 窓サイズをまたいだ統合は試されましたか

なぜ効くか ── 論文は「窓サイズの異なる確率マップどうしは統合していない」と明記し、将来の切り分けとして自ら挙げている。 224は再現率が高く（0.70）、448・672は適合率が高い（0.78）ので、統合すれば両取りできる可能性がある。前向きな質問。

#### 日本語

表IIIで、224は再現率が高く、448と 672は適合率が高い、という違いがありました。この3つの確率マップを統合することは試されましたか。 論文では「統合していない」と書かれていましたが、統合すると再現率と適合率のどちらに寄ると予想されますか。

#### English

In Table III, the 224 window gives the highest recall while 448 and 672 give higher precision. Have you tried combining the probability maps from the three window sizes? The paper says you did not — if you did, would you expect the result to move toward recall or toward precision?

### 第3群：用途と運用（防災科研の方に向く）

### Q8 ★★☆ 画素ではなく、崩壊地の箇所数で測るとどうなりますか

なぜ効くか ── 論文の評価は画素単位のみ。しかし結論は「複雑な山地での実運用にはまだ足りない」と判断している。その判断は画素の話であって、「何箇所見つけたか」には答えていない。 現場が問うのは箇所数のはず、という問題提起。

#### 日本語

評価は画素単位の再現率・適合率で行われていました。実際に使う場面では「崩壊地を何箇所見つけられたか」のほうが重要になることはありませんか。 たとえば連結成分ごとに対応づけて数える評価をされると、数字はどう変わりそうでしょうか。

#### English

Your evaluation is pixel-wise. In actual use, would "how many landslides were found" not matter more than how many pixels were coloured? If you matched connected components and counted detections per landslide, how do you think the numbers would change?

### Q9 ★★☆ この用途では、見逃しと誤報のどちらのコストが高いのですか

なぜ効くか ── 表Iで、3クラスは適合率寄り（0.76/0.64）、2クラスは再現率寄り（0.73/0.68）で、F値は同じ 0.70。 論文自身が「どちらも他方を支配しない。運用上のコストの比で選ぶべき」と書いている。ならばその比はいくつなのかを聞くのは自然な続き。災害後の応急対応という文脈に踏み込める。

#### 日本語

表Iで、2クラスと3クラスは F値が同じ 0.70で、動作点だけが違うと書かれていました。そして「運用上のコストの比で選ぶべき」とも。災害直後の応急対応という用途では、見逃しと誤報のコストの比はどのくらいだとお考えですか。 どちらに寄せるのが適切でしょうか。

#### English

In Table I, the binary and three-class settings give the same F1 of 0.70 and differ only in the operating point, and you write that the choice should follow the relative operational costs. For

emergency response immediately after a disaster, what do you think that cost ratio actually is? Which way should the operating point be set?

### Q10 ★☆☆ 検証データはどのように分けられましたか

なぜ効くか ── 学習・検証は同じ12組の画像から取られている。テストは独立4地域で正しく分けているが、検証タイルが学習タイルの隣にある可能性があり、その検証で最良エポックを選んでいる。地理的な分割をしたかを聞くだけで、咎めにはならない。

#### 日本語

テストは学習に含まれない独立4地域とのことで、そこは明確でした。検証データは、学習に使った12組の画像からどのように分けられましたか。 無作為でしょうか、それとも地理的に分けられましたか。

#### English

The test set is four independent regions not used in training — that was clear. How was the validation set separated from the twelve training image pairs — randomly, or geographically?

### Q11 ★☆☆ ストライドをもっと細かくすると、どこまで効きますか

なぜ効くか ── いまは窓の半分、1画素あたり最大4タイル。もっと重ねれば再現率はさらに上がるのか、それとも頭打ちになるのか。 計算コストとの交換になるので、相手が測っていない「コスト」の話にも自然につながる。

#### 日本語

ストライドは窓の半分で、1画素あたり最大4つのタイルから予測される、とのことでした。ストライドをさらに細かくすると、再現率はどこまで伸びますか。 頭打ちになる点はありそうでしょうか。

#### English

Your stride is half the window, so each pixel is predicted by up to four tiles. If you made the stride finer, how far would recall keep improving? Would you expect it to saturate at some point?

### Q12 ★★★ 単一スケール2クラスの適合率 0.28は、公正な比較対象ですか

最も踏み込む質問。使うなら最後の懇親会で、口調をやわらかく。なぜ効くか ── 表Iで、単一スケール2クラスだけ適合率が 0.28と極端に低く、ここからの改善（0.28 → 0.73）が「マルチスケール学習の効果」として報告されている。ベースラインが壊れていた可能性を排除できないと、+0.30の大きさは信用しづらい。

#### 日本語

表Iの単一スケール2クラスですが、適合率 0.28は他の3条件（0.68〜0.76）と比べてかなり低い値だと感じました。この条件だけ学習が十分に進んでいなかった、ということはないでしょうか。 学習曲線はほかの条件と同じような形でしたか。

#### English

About the single-scale binary row in Table I — a precision of 0.28 is a long way below the other three conditions, which are 0.68 to 0.76. Is it possible that this one condition simply did not converge as well? Did its training curve look similar to the others?

言い方に注意 ── 「おかしい」ではなく「気になった」と言うこと。English: use "I was curious about …" rather than "That looks wrong."

### 迷ったときの1問

質疑が1問しかなく、どれにするか決められないなら **Q3（ふつうの U-Net で試したか）**。理由：

- 相手の論文が自分で立てた前提の、自然な続きになっている（攻撃にならない）
- 相手にとって次の実験の示唆になる（価値を渡せる）
- 返ってきた答えが、そのまま賀屋さんの研究の立ち位置の説明につながる
- 聴衆にも分かりやすい（専門用語が要らない）

第2候補は Q1（ばらつき）。 ただしこれは論文の弱点を突くので、Q3のあとの2問目に回すほうが場の流れとしては良い。

### 読み上げ用（短縮版）

日本語（そのまま言える長さ）

#### English

| | 日本語（そのまま言える長さ）| English |
|---|---|---|
| **Q1** | 「同じ条件を乱数の種だけ変えて学習し直すと、F値はどのくらいばらつきますか。その幅と比べて +0.02はどの程度でしょうか」 | "If you retrain the same configuration with a different random seed, how much does F1 move? How does +0.02 compare with that spread?" |
| **Q2** | 「融合の前の確率の分布は、0と1の両端に寄っていましたか、中間にも分布していましたか」 | "Before fusion, is the probability distribution concentrated near 0 and 1, or is there mass in between?" |
| **Q3** | 「同じ学習方法と同じ融合を、ふつうの U-Netに適用されたことはありますか。差はどのくらいでしたか」 | "Have you applied the same training recipe and the same fusion to a plain U-Net? How large was the difference?" |
| **Q4** | 「傾斜や曲率を加えるとき、写真と同じ符号化器に重ねますか、別系統にしますか」 | "When you add slope and curvature, would you stack them into the same encoder as the photograph, or use a separate encoder?" |
| **Q5** | 「新しい崩壊痕も裸地です。判読の段階でこの2つはどう区別されましたか」 | "A fresh landslide scar is also bare ground. How were the two distinguished at the interpretation stage?" |
| **Q8** | 「画素ではなく、崩壊地の箇所数で測ると数字はどう変わりそうでしょうか」 | "If you counted detections per landslide instead of per pixel, how would the numbers change?" |
