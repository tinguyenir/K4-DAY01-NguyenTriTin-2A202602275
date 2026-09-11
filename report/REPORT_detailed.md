# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python `3.13.15` / PyTorch `2.11.0+cpu` / Ultralytics `8.4.145`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không. Giữ nguyên ba checkpoint, taxonomy, dữ liệu mẫu và ngưỡng mặc định `0.35` cho detection/instance segmentation. Không huấn luyện hoặc fine-tune model.

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

---

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`; hình đối chiếu: `visuals/classification_top5.png`.

### 1.1. Record hạng 1

- `sample_id = "traffic"`
- `class_id = 468`
- `class_name = "cab"`
- `rank = 1`
- `score = 0.510915`
- `taxonomy_name = "ImageNet-1K"`
- `model_file = "yolo11n-cls.pt"`
- Kích thước ảnh: `640 × 428` pixel.

Top-5 của sample `traffic` là: 1. `cab` (0.510915), 2. `minibus` (0.164284), 3. `police_van` (0.085848), 4. `recreational_vehicle` (0.054110), 5. `streetcar` (0.048193).

### 1.2. Record này mô tả toàn ảnh như thế nào?

Đây là **prediction cấp ảnh**. Model không tạo một box riêng cho từng phương tiện mà nhìn toàn bộ ảnh `traffic` rồi xếp hạng các lớp ImageNet-1K. Lớp `cab` đứng hạng 1 với score `0.510915`. Hình minh họa cho thấy ảnh có nhiều phương tiện gồm xe buýt, ô tô và các phương tiện khác, vì vậy nhãn hạng 1 chỉ phản ánh lớp mà model cho là phù hợp nhất với toàn ảnh chứ không mô tả đầy đủ tất cả object xuất hiện.

Điểm cần phân biệt là:

- **Classification:** một prediction mô tả toàn ảnh.
- **Detection:** một prediction mô tả một object và vị trí của object.
- **Instance segmentation:** một prediction mô tả một instance cùng vùng mask/polygon.

### 1.3. Ai định nghĩa class list mà checkpoint có thể dự đoán?

Class list được xác định bởi **taxonomy/dataset gắn với checkpoint**, không phải do model tự nghĩ ra khi inference. Với `yolo11n-cls.pt`, trường `taxonomy_name` trong JSON là `ImageNet-1K`, do đó `class_id=468` và tên `cab` phải được hiểu trong taxonomy ImageNet-1K.

Nói cách khác, quá trình inference chỉ chọn/xếp hạng trong tập lớp đã biết. Nếu dự án thực tế cần taxonomy khác thì cần xây dựng mapping, huấn luyện/fine-tune checkpoint phù hợp hoặc thay model; không thể tự diễn giải ID cũ sang taxonomy mới mà không có quy tắc.

### 1.4. Vì sao cần giữ cả ID, tên lớp và tên taxonomy?

Ba trường có vai trò khác nhau:

- `class_id`: định danh số, thuận tiện cho code, database và tính metric.
- `class_name`: tên dễ đọc cho annotator/reviewer.
- `taxonomy_name`: cho biết ID và tên lớp thuộc hệ nhãn nào.

Ví dụ `class_id=468` chỉ có ý nghĩa đầy đủ khi biết nó thuộc `ImageNet-1K`. Nếu chỉ lưu ID mà mất taxonomy, việc tái lập hoặc đối chiếu dữ liệu sau này có thể bị sai.

### 1.5. Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?

Guideline cần quy định rõ bài toán là **single-label** hay **multi-label**.

Nếu là single-label, guideline nên nêu rõ:

1. Tiêu chí chọn lớp chính: object chiếm diện tích lớn nhất, object trung tâm, object theo mục tiêu nghiệp vụ hay ngữ cảnh toàn ảnh.
2. Khi nhiều chủ thể quan trọng tương đương thì ưu tiên lớp nào.
3. Khi ảnh không có một chủ thể trội thì có dùng lớp `other/background/uncertain` hay không.
4. Khi nào annotator phải escalation thay vì tự chọn theo cảm tính.
5. Có được suy đoán object bị che khuất hoặc không nhìn rõ hay không.

Ảnh `traffic` là ví dụ phù hợp để cho thấy nếu guideline không rõ, hai annotator có thể chọn nhãn khác nhau dù cùng nhìn một ảnh.

### 1.6. Vì sao model score không phải ground truth?

`score=0.510915` là **độ tin cậy của model đối với prediction**, không phải độ đúng của nhãn và cũng không phải điểm chất lượng ground truth.

Ground truth phải được tạo bởi annotator theo guideline và được reviewer/QC xác nhận. Model có thể dự đoán sai dù score cao và cũng có thể dự đoán đúng với score thấp. Vì vậy không được sao chép prediction thành ground truth chỉ dựa trên confidence.

### 1.7. Quan sát và kết luận cho classification

Kết quả cho thấy top-1 chỉ đạt khoảng `0.511`, trong khi các hạng sau giảm nhanh (`minibus=0.164284`, `police_van=0.085848`, ...). Điều này cho thấy model chưa có một lựa chọn áp đảo tuyệt đối cho ảnh nhiều phương tiện. Đây là bằng chứng tốt để nhấn mạnh rằng **model prediction là dữ liệu cần kiểm tra**, không phải nhãn chuẩn.

---

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

### 2.1. Một record cụ thể

Chọn prediction có score cao nhất trong sample `kitchen`:

- `class_name = "person"`
- `score = 0.912625`
- `bbox_xyxy = [385.33, 69.24, 498.92, 348.92]`
- `bbox_width = 113.58 px`
- `bbox_height = 279.68 px`
- `bbox_format = "xyxy"`
- `coordinate_unit = "pixel"`
- `score_threshold = 0.35`
- Kích thước ảnh: `640 × 427` pixel.

Trong định dạng `xyxy`:

`[x_min, y_min, x_max, y_max] = [385.33, 69.24, 498.92, 348.92]`.

### 2.2. Diễn giải vị trí box bằng lời

Gốc tọa độ nằm ở góc trên trái ảnh; trục `x` tăng từ trái sang phải và `y` tăng từ trên xuống dưới.

Box của `person`:

- bắt đầu tại `(x_min, y_min) = (385.33, 69.24)`;
- kết thúc tại `(x_max, y_max) = (498.92, 348.92)`;
- tâm box, **tính từ tọa độ JSON**, khoảng `(442.12, 209.08)`;
- chiều rộng chiếm khoảng `17.7%` chiều rộng ảnh;
- chiều cao chiếm khoảng `65.5%` chiều cao ảnh;
- diện tích hình chữ nhật box chiếm khoảng `11.6%` diện tích ảnh.

Đối chiếu hình `detection_predictions.png`, box này bao quanh người đứng ở vùng giữa-phải của bức ảnh bếp. Box cao và tương đối hẹp, phù hợp với hình dạng một người đứng.

### 2.3. Các prediction trong sample `kitchen`

Ở threshold mặc định `0.35`, JSON lưu **11 prediction**:

- `person`: 2 prediction;
- `bowl`: 5 prediction;
- `oven`: 2 prediction;
- `cup`: 2 prediction.

Các score trải từ `0.381215` đến `0.912625`.

Điều này cho thấy một ảnh có thể tạo nhiều record; mỗi record tương ứng với **một object được model phát hiện**, kể cả khi nhiều object cùng class, ví dụ nhiều `bowl`.

### 2.4. So sánh số prediction ở hai threshold

- Với `threshold=0.35`: có **11 prediction**.
- Nếu dùng `threshold=0.60`: có **6 prediction** có score `>= 0.60`.

Như vậy khi tăng threshold từ `0.35` lên `0.60`, số prediction giảm **5 object**, tương đương giảm khoảng `45.5%` so với tập prediction ở threshold `0.35`.

**Lưu ý về nguồn số liệu:** số `11` lấy trực tiếp từ các record `kitchen` trong `detection_predictions.json`. Số `6` là giá trị **tính toán** bằng cách đếm các record đó có `score >= 0.60`. Tôi không suy đoán số prediction ở threshold `0.20` vì file JSON được lưu ở threshold `0.35` không chứa các prediction đã bị loại bên dưới `0.35`.

### 2.5. Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?

Khi threshold thấp:

- model giữ nhiều candidate hơn;
- khả năng bao phủ object thật thường tăng;
- reviewer phải kiểm tra nhiều prediction hơn;
- số false positive có thể tăng.

Khi threshold cao:

- model loại nhiều prediction score thấp;
- reviewer có ít box hơn để kiểm tra;
- false positive có thể giảm;
- nhưng nguy cơ bỏ sót object thật tăng.

Vì vậy threshold là **cấu hình lọc prediction**, không phải quy tắc gán ground truth. Annotator vẫn phải gán nhãn object thuộc scope ngay cả khi model không phát hiện.

### 2.6. Đề xuất một quy tắc box chặt

Một quy tắc có thể áp dụng nhất quán:

> Vẽ bounding box là hình chữ nhật nhỏ nhất bao phủ toàn bộ **phần object nhìn thấy** thuộc phạm vi gán nhãn. Bốn cạnh box phải bám gần các điểm cực trái, phải, trên và dưới của object, không để khoảng nền thừa rõ rệt. Không lấy bóng đổ hoặc object lân cận vào box trừ khi guideline định nghĩa chúng là một phần của object.

Đối với object bị cắt bởi biên ảnh, box kết thúc tại biên ảnh. Không tự kéo box ra ngoài ảnh.

### 2.7. Object bị che khuất/cắt mép cần guideline hoặc escalation quyết định gì?

Guideline cần trả lời tối thiểu các câu hỏi:

1. Chỉ cần nhìn thấy bao nhiêu phần trăm object thì vẫn gán nhãn?
2. Box bao phần **visible** hay ước lượng toàn bộ object (**amodal**)?
3. Có cần thuộc tính `occluded` hoặc `truncated` không?
4. Object rất nhỏ hoặc chỉ hiện một phần có được giữ không?
5. Nếu class không chắc chắn thì chọn `unknown`, bỏ qua hay escalation?

Trong hình `kitchen`, prediction `person` ở mép trái chỉ nhìn thấy một phần cơ thể và bị cắt bởi biên ảnh. Đây là ví dụ điển hình mà guideline phải quy định rõ; annotator không nên tự suy đoán phần cơ thể nằm ngoài ảnh.

### 2.8. Điểm mơ hồ quan sát được và QC

Một số box `bowl`/`cup` nhỏ nằm gần nhau trên bàn. Trong tình huống này có thể phát sinh:

- box trùng/duplicate;
- nhầm `bowl` với `cup`;
- bỏ sót object nhỏ;
- box lấy quá nhiều nền;
- box gộp hai object thành một.

Reviewer cần đối chiếu trực tiếp ảnh, không chỉ nhìn JSON. Các prediction score thấp như khoảng `0.38–0.50` đáng được kiểm tra kỹ nhưng không được tự động coi là sai.

---

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

### 3.1. Một record cụ thể

Chọn instance có score cao nhất:

- `instance_id = "kitchen-001"`
- `class_name = "person"`
- `score = 0.899318`
- `bbox_xyxy = [385.45, 66.44, 498.02, 348.58]`
- `polygon_point_count = 348`
- `score_threshold = 0.35`
- `coordinate_unit = "pixel"`

Mười điểm đầu của `polygon_xy`:

`[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], [441.0, 73.0], [439.0, 73.0], [438.0, 74.0], [437.0, 74.0], [436.0, 75.0], ...`

Tổng cộng polygon có **348 điểm**.

### 3.2. Polygon bổ sung chi tiết gì so với box?

Bounding box chỉ cho biết một hình chữ nhật bao quanh object. Trong hình chữ nhật đó thường vẫn có nhiều pixel nền.

Polygon đi theo đường biên object nên mô tả được:

- hình dạng thực của object;
- các vùng lồi/lõm;
- phần nền cần loại bỏ;
- ranh giới giữa các instance gần nhau;
- diện tích pixel gần với object thật hơn box.

Trong hình segmentation, mask `person` bám vào đầu, thân, tay và chân thay vì tô toàn bộ hình chữ nhật như detection box. Đây là khác biệt quan trọng giữa detection và instance segmentation.

### 3.3. `instance_id` dùng để làm gì và không phải loại ID nào?

`instance_id` dùng để phân biệt **từng object riêng trong một output**, ví dụ:

- `kitchen-001`: một `person`;
- các `bowl` khác nhau sẽ có các `instance_id` khác nhau dù cùng `class_name="bowl"`.

`instance_id`:

- không phải `class_id`;
- không phải ID taxonomy;
- không phải tracking ID để nhận diện cùng một vật thể qua nhiều frame/video.

Trong notebook, ID được tạo theo dạng `<sample_id>-NNN`, mục đích là định danh duy nhất từng instance trong output lab.

### 3.4. Các instance quan sát được trong `kitchen`

Ở threshold `0.35`, file segmentation có **11 instance**:

- `person`: 2;
- `bowl`: 4;
- `dining table`: 1;
- `oven`: 1;
- `potted plant`: 1;
- `spoon`: 2.

So với detection, danh sách class không hoàn toàn giống nhau. Điều này không có nghĩa một file là ground truth; hai checkpoint đang tạo **prediction riêng** cho hai tác vụ khác nhau.

### 3.5. Đề xuất một quy tắc biên mask

Quy tắc đề xuất:

> Mask phải bám theo biên của **phần object nhìn thấy**. Không để mask tràn rõ rệt sang nền hoặc object khác. Không tự nội suy phần bị che khuất hoặc nằm ngoài ảnh nếu guideline không yêu cầu amodal segmentation. Các lỗ, khe và vùng bên trong object chỉ được giữ/loại theo cùng một quy tắc áp dụng cho toàn dataset.

Với pixel biên có anti-aliasing hoặc biên mờ, guideline cần quy định annotator ưu tiên biên hình học quan sát được ở mức zoom phù hợp, thay vì mỗi người tự chọn một ngưỡng khác nhau.

### 3.6. Vùng mờ/tiếp xúc/che khuất cần guideline hoặc escalation quyết định gì?

Guideline cần làm rõ:

1. Hai object cùng lớp chạm nhau có tách thành hai instance không?
2. Biên giữa hai object gần cùng màu được xác định bằng tín hiệu nào?
3. Vùng bị object khác che có được suy đoán hay chỉ mask phần nhìn thấy?
4. Object quá nhỏ có cần polygon hay được bỏ theo ngưỡng kích thước?
5. Khi không xác định được biên chính xác thì annotator được phép dùng mức xấp xỉ nào?
6. Khi nào bắt buộc chuyển reviewer/lead quyết định?

Nếu không có rule rõ, hai annotator dễ tạo mask khác nhau cho cùng một ảnh, làm giảm consistency của ground truth.

### 3.7. Điểm mơ hồ quan sát được và QC

Sample `kitchen` chứa nhiều object nhỏ/chồng lấp: bát, thìa, khu vực bàn, người và các vật treo. Vì vậy segmentation khó hơn detection ở chỗ reviewer phải kiểm tra **chất lượng biên**, không chỉ class và vị trí.

Reviewer nên kiểm tra:

- mask có tràn vào nền không (`leakage`);
- mask có bỏ thiếu vùng object không (`under-segmentation`);
- hai object có bị gộp thành một mask không (`merge`);
- một object có bị chia thành nhiều instance không cần thiết không (`split`);
- instance cùng lớp có ID riêng không;
- polygon có bám biên nhất quán không.

---

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

### 4.1. Ý nghĩa từng bước

1. **Ảnh thô:** dữ liệu đầu vào chưa có nhãn.
2. **Guideline:** tài liệu quy định taxonomy, phạm vi object, cách vẽ box/mask và cách xử lý trường hợp mơ hồ.
3. **Ground truth:** nhãn do con người tạo theo guideline; đây là dữ liệu chuẩn dùng cho training/evaluation.
4. **Huấn luyện:** model học từ dữ liệu đã được gán nhãn.
5. **Prediction:** đầu ra của model trên ảnh mới; có thể đúng hoặc sai.
6. **QC/rework:** reviewer kiểm tra tính đúng/nhất quán; annotation sai hoặc chưa rõ được trả lại để sửa.

### 4.2. Bảng tổng hợp

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một hoặc nhiều class ở cấp ảnh theo taxonomy dự án; lưu ID/tên lớp và taxonomy | Ảnh `traffic` có nhiều loại phương tiện nên khó chọn một lớp duy nhất nếu rule không rõ | Đọc toàn ảnh, áp dụng đúng single-label/multi-label; chọn lớp theo guideline; đánh dấu uncertain/escalate khi cần | Kiểm tra class có thuộc taxonomy; nhãn có đúng scope; các ảnh nhiều chủ thể có được xử lý nhất quán |
| Phát hiện vật thể | Mỗi object là một record gồm class + bounding box, ví dụ `xyxy` pixel | Object nhỏ, overlap, occluded, truncated; `person` ở mép trái ảnh `kitchen` là ví dụ cắt mép | Gán tất cả object thuộc scope; vẽ box chặt; không bỏ object chỉ vì model không phát hiện; escalation ca mơ hồ | Kiểm tra sai class, missed object, duplicate, box quá rộng/quá chặt, xử lý occlusion/truncation |
| Instance segmentation | Mỗi object là một instance gồm class + polygon/mask; instance có ID riêng | Biên mờ, object tiếp xúc/chồng lấp, vùng nhỏ làm đường biên khó xác định | Tách đúng instance; vẽ mask theo rule; không suy đoán vùng ẩn nếu guideline không yêu cầu | Kiểm tra leakage, under-segmentation, merge/split, biên không nhất quán, instance ID và class |

### 4.3. Phân biệt lỗi prediction và lỗi ground truth

Một prediction sai không đồng nghĩa ground truth sai. Hai loại artifact có vai trò khác nhau:

- Nếu model bỏ sót object thật: đó là **false negative của prediction**; annotator vẫn phải gán object đó nếu guideline yêu cầu.
- Nếu model tạo box cho object không tồn tại: đó có thể là **false positive của prediction**.
- Nếu annotator vẽ box sai hoặc gán sai class: đó là **annotation error**, phải qua QC/rework.
- Nếu guideline không nói rõ cách xử lý một ca: đây là **guideline ambiguity**, không nên ép annotator tự đoán.

### 4.4. Các kiểm tra QC nên thực hiện

QC nên kiểm tra cả **schema** và **nội dung thị giác**:

- đủ trường bắt buộc;
- class ID khớp class name/taxonomy;
- score nằm trong `[0,1]` đối với prediction;
- box có `x_min < x_max`, `y_min < y_max` và nằm trong ảnh;
- polygon có ít nhất 3 điểm và nằm trong kích thước ảnh;
- `instance_id` không trùng;
- visual overlay khớp JSON;
- không có missed/duplicate object đáng chú ý;
- cách xử lý ca mơ hồ nhất quán với guideline.

---

## 5. An toàn dữ liệu

### 5.1. Một quy tắc bảo vệ dữ liệu

Chỉ sử dụng dữ liệu đã được cấp phép và đúng phạm vi của bài thực hành. Không đưa ảnh cá nhân, thông tin định danh cá nhân, dữ liệu khách hàng, dữ liệu nội bộ hoặc dữ liệu nhạy cảm vào notebook, output hoặc repository công khai.

Các ảnh của bài lab là ảnh COCO 2017 validation công khai; file `IMAGE_ATTRIBUTION.md` lưu nguồn, tác giả và giấy phép cho từng sample.

### 5.2. Khi gặp dữ liệu không đúng phạm vi

Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ:

1. dừng xử lý;
2. không upload/commit/chia sẻ thêm;
3. giữ nguyên evidence cần thiết nhưng không sao chép dữ liệu sang nơi khác;
4. báo cho **Lab Coach/mentor phụ trách** để xác nhận cách xử lý;
5. chỉ tiếp tục khi nhận được hướng dẫn phù hợp.

---

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.

### 6.1. Provenance chính để tái lập

- Classification checkpoint: `yolo11n-cls.pt`
- Detection checkpoint: `yolo11n.pt`
- Segmentation checkpoint: `yolo11n-seg.pt`
- Ultralytics: `8.4.145`
- Classification taxonomy: `ImageNet-1K`
- Detection/segmentation taxonomy: `COCO-80`
- Detection threshold mặc định: `0.35`
- Segmentation threshold mặc định: `0.35`
- Runtime đã chạy: CPU
- Sample chính dùng trong báo cáo:
  - `traffic`, COCO image ID `210273`;
  - `kitchen`, COCO image ID `397133`.

### 6.2. Kết luận

Qua ba tác vụ, tôi phân biệt được ba mức biểu diễn nhãn:

- **Classification:** class ở cấp toàn ảnh.
- **Detection:** class + bounding box cho từng object.
- **Instance segmentation:** class + vùng mask/polygon cho từng instance.

Điểm quan trọng nhất là **prediction không phải ground truth**. Model score chỉ phản ánh độ tin cậy của model. Ground truth phải được tạo theo guideline, kiểm tra bằng QC và rework khi cần. Threshold có thể thay đổi số prediction mà reviewer nhìn thấy, nhưng không được dùng để quyết định object nào tồn tại trong ground truth.
