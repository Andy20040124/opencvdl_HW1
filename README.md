# import photo 
目前的話只有load_image_1 有實際的使用意義，load_image_2還沒有進行功能的賦予
load_image_1 會覆蓋先前輸入進來的圖片，並且有中文的檔名會有錯誤產生，因此**請使用英文、數字、其他字符進行檔案命名**
## 防呆機制
因為在還沒上傳圖片的時候，若點擊後面的按鍵會出現錯誤訊息，因此若沒有上傳圖片之前，點擊後面的圖片會出現 "photo haven't been uploaded"

# Q1
## color separation 
上傳照片後按下按鍵可以生成RGB照片顏色分解的三色通道照片各一張
## Color Transformation
上傳照片之後生成original grayscale, 另一張照片為avg  grayscale,因為 (b/3 + g/3 + r/3)算出來的值跟(r+g+b)/3不同，因此我採取後者的計算方式來進行
## Color Extraction
上傳照片之後利用篩選色彩區間來把黃色以及綠色選出來當作mask，題目給的range差不多是15\~85，但是我應該會比較希望去的更乾淨一點，因此往外取了10\~ 113的range，因此來達到更乾淨的紅藍圖

# Q2
## Gaussian blur
透過呼叫**cv2.createTrackbar('trackbar name', 'window's name', min, max, fn)**讓他跟imshow的照片生成在同一個pop window 當中，並且利用**cv2.setTrackbarPos('trackbar name', 'window's name')**來反應當下拉動的值並且回饋到圖片上
## Bilateral Filter
程序基本上跟gaussian fliter 一樣，修改kernal為圖片的diameter
## Median Filter
程序基本上跟gaussian fliter 一樣，修改kernal為圖片的diameter

# Q3
## Sobel X
透過先轉換成灰階後取出normalized之後的值，並且執行smoothing後選擇**cv2.Sobel(blur,-1,1,0)**，也就是保留垂直的資料
## Sobel Y
透過先轉換成灰階後取出normalized之後的值，並且執行smoothing後選擇**cv2.Sobel(blur,-1,0,1)**，也就是保留水平的資料
## Combination and Threshold 
這邊是主要遇到問題的地方，cv2.Sobel不論是使用cv2.CV_64F或者是-1來進行取值，得到的square root 圖都沒有辦法獲得跟測試影片一樣乾淨的圖片，但是在針對threshold=128 & 28 的時候，可以很明顯的看出有過濾到部分的資料，代表threshold還是有發揮其作用
## grafient angle 
先使用sobel x & y 來進行梯度的計算，並且利用角度計算的公式(from CSDN) 然後因為arctan 值域在180度內，若要進行整個圓的轉換的話就要使用arctan2來進行轉換，然後把所有負角度的全部+360讓他轉回來就可以開始生成mask，最後使用inrange把mask 生成然後跟magnitude來進行bitwise_and運算即可

# Q4
## transform
先把所有需要的資料從UI當中利用.text()抓進來使用
再來先把角度從百分比轉換成rad，最後把rotation 需要的資料全部都放進去matrix(透過把旋轉、縮放以及位移矩陣同時結合計算出一個大的rotation matrix)，並且利用錯誤訊息(M0.rows == 2 && M0.cols == 3)來得知需要把矩陣縮減成 row=2 , col=3 的狀態再去進行warpaffine，最後得到解答