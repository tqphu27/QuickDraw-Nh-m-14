# Mạng tích chập (CNN)
1. Giới thiệu tích chập
 Tích chập là một khái niệm trong xử lí tín hiệu số nhằm biến đổi thông tin đầu vào thông qua một phép tích chập với bộ lọc để trả về đầu ra là một tín hiệu mới. Tín hiệu này sẽ làm giảm những đặc trưng mà bộ lọc không quan tâm và chỉ giữ những đặc trưng chính.
  
 Tích chập được ứng dụng phổ biến trong lĩnh vực thị giác máy tính. Thông qua các phép tích chập, các đặc trưng chính từ ảnh được chiết xuất và truyền vào các lớp tích chập (layer convolution). Mỗi một lớp tích chập sẽ bao gồm nhiều đơn vị mà kết quả ở mỗi đơn vị là một phép biến đổi tích chập từ layer trước đó thông qua phép nhân tích chập với bộ lọc.
 
 Về cơ bản thiết kế của một mạng nơ ron tích chập 2 chiều có dạng như sau:
       INPUT -> [[CONV -> RELU]N -> POOL]M -> [FC -> RELU]*K -> FC
       
       Trong đó:
         INPUT: Lớp đầu vào
         CONV: Lớp tích chập
         RELU: Lớp biến đổi thông qua hàm kích hoạt relu để kích hoạt tính phi tuyến
         POOL: Lớp tổng hợp, thông thường là Max pooling hoặc có thể là Average pooling dùng để giảm chiều của ma                      trận đầu vào.
         FC: Lớp kết nối hoàn toàn. Thông thường lớp này nằm ở sau cùng và kết nối với các đơn vị đại diện cho nhóm                  phân loại.
 Quá trình:
   Quá trình chiết xuất đặc trưng: Thông qua các tích chập giữa ma trần đầu vào với bộ lọc để tạo thành các đơn vị trong một lớp mới. Quá trình này có thể diễn ra liên tục ở phần đầu của mạng và thường sử dụng hàm kích hoạt relu
   Quá trình tổng hợp: Các lớp ở về sau quá trình chiết xuất đặc trưng sẽ có kích thước lớn do số đơn vị ở các lớp sau thường tăng tiến theo cấp số nhân. Điều đó làm tăng số lượng hệ số và khối lượng tính toán trong mạng nơ ron.
    => Để giảm tải tính toán => giảm chiều ỏ giảm số đơn vị của lớp. 
    (Vì mỗi đơn vị sẽ là kết quả đại diện của việc áp dụng một bộ lọc để tìm ra một đặc trưng => không khả thi.)
    => Giảm chiều. 
   Quá trình kết nối: Sau khi đã giảm số lượng tham số đến một mức độ hợp lý
    => ma trận cần được làm dẹt(flatten) thành 1 vector (Quá trình này diễn ra cuối mạng tích chập và sử dụng reLu)
    => Kết nối cuối cùng sẽ dẫn tới các đơn vị là đại diện cho mỗi lớp với hàm kích hoạt softmax nhằm mục đích tích xác xuất.
   <img src= "https://developer.apple.com/library/archive/documentation/Performance/Conceptual/vImage/Art/kernel_convolution.jpg">
   
  <img src = "https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed.gif">
   
   
# Quick Draw

Có 3 file chính trong project.
 
- Load data + Trainer sẽ lẽ file mô phỏng ngắn gọn về việc xử lý dữ liệu và traning
([Notebook](https://colab.research.google.com/drive/1aSfciE9msnYFWKnUkaFowD0npnLdxkM1#scrollTo=aR2Ws8jiJA3-))
- File App: mô phỏng lại kết quả

- Kết quả: [Watch the video](https://drive.google.com/file/d/18JTq-_9eOeDBeqKbQY0za9Ccrc_XhHQS/view?usp=sharing)
