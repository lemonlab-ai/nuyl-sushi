<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  color: #e0e0e0;
  font-size: 16px;
}

body {
  background: #1a1a1a;
}

main {
  width: 90vw;
  max-width: 900px;
  height: 80vh;
  display: flex;
  justify-content: center;
  align-items: flex-start;
}
.content {
  width: 100%;
}
.input-item {
  width: 100%;
  height: 60px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #404040;
  border-radius: 0 10px 10px 0;
  margin-bottom: 30px;
}

.input-item-left {
  display: flex;
  align-items: center;
  height: 100%;
}

.pink-line {
  width: 5px;
  height: 100%;
  background: #ff4081;
}

.blue-line {
  width: 5px;
  height: 100%;
  background: #00e5ff;
}

.status {
  margin-left: 10px;
}

.btn {
  background: #ff4081;
  padding: 10px 14px;
  margin-right: 20px;
  border-radius: 10px;
  cursor: pointer;
  transition: background 0.3s ease-in-out;
}

.btn:hover {
  background: #ff79b0;
}

.disable {
  background: #e0e0e0;
  color: #404040;
}

.disable:hover {
  background: #b5b5b5;
  color: #404040;
}

.hidden {
  display: none;
}

.video-content {
  width: 100%;
  height: 100%;
}
.videoPlayer {
  width: 100%;
  background-size: contain;
}
</style>

<template>
  <main>
    <div class="content">
      <div class="input-item">
        <div class="input-item-left">
          <div :class="[isApiConnected ? 'blue-line' : 'pink-line']"></div>
          <div class="status">
            api連線狀態：{{ isApiConnected ? "已連線" : "連線失敗" }}
          </div>
        </div>
        <div class="input-item-right" :class="[isApiConnected ? 'hidden' : '']">
          <label class="btn btn-info">
            <input
              style="display: none"
              @click="connectToApi()"
              type="button"
            />
            <i></i> {{ isApiConnected ? "已連線" : "重新連線" }}
          </label>
        </div>
      </div>

      <div class="input-item">
        <div class="input-item-left">
          <div :class="[isFileSelected ? 'blue-line' : 'pink-line']"></div>
          <div class="status">檔案選擇：{{ fileSelectStatus }}</div>
        </div>
        <div class="input-item-right">
          <label class="btn btn-info">
            <input
              style="display: none"
              @change="onFileSelect"
              type="file"
              accept="video/.mp4"
            />
            <i></i> {{ isFileSelected ? "重新選擇" : "選擇檔案" }}
          </label>
        </div>
      </div>

      <div class="input-item">
        <div class="input-item-left">
          <div
            :class="[uploadStatus == '上傳成功' ? 'blue-line' : 'pink-line']"
          ></div>
          <div class="status">上傳狀態：{{ uploadStatus }}</div>
        </div>
        <div class="input-item-right">
          <label class="btn btn-info" :class="uploadBtnColor">
            <input style="display: none" @click="onUpload()" type="button" />
            <i></i> {{ uploadBtn }}
          </label>
        </div>
      </div>
      <div class="video-content">
        <video controls class="videoPlayer" id="videoPlayer">
          <source src="" type="video/mp4" />
        </video>
      </div>
    </div>
  </main>
</template>

<script>
import axios from "axios";
export default {
  name: "FileSelecter",
  mounted() {
    this.connectToApi();
  },
  data() {
    return {
      file: "",
      isApiConnected: false,
      isFileSelected: false,
      isFormatWrong: false,
      uploadStatus: "尚未上傳",
      fileName: "",
      currentFile: "",
    };
  },
  computed: {
    fileSelectStatus() {
      if (!this.isFileSelected && !this.isFormatWrong) {
        return "尚未選擇";
      } else if (this.isFormatWrong) {
        return "請選擇mp4格式的檔案";
      } else {
        return this.fileName;
      }
    },
    uploadBtn() {
      if (
        this.uploadStatus == "上傳成功" &&
        this.currentFile != this.fileName &&
        this.isFileSelected
      ) {
        return "再次上傳";
      } else if (this.uploadStatus == "上傳失敗") {
        return "重新上傳";
      } else if (!this.isApiConnected) {
        return "尚未連線";
      } else if (!this.isFileSelected) {
        return "尚未選擇檔案";
      } else if (this.currentFile == this.fileName) {
        return "已上傳";
      } else if (this.uploadStatus == "上傳中") {
        return "上傳中";
      } else {
        return "上傳檔案";
      }
    },
    uploadBtnColor() {
      if (
        !this.isApiConnected ||
        !this.isFileSelected ||
        this.uploadStatus == "上傳中" ||
        this.currentFile == this.fileName
      ) {
        return ["disable"];
      } else {
        return [];
      }
    },
  },
  methods: {
    connectToApi() {
      axios
        .get("http://localhost:8000/check_connection/")
        .then((res) => {
          const connectionStatus = res.data.connected;
          if (connectionStatus) {
            this.isApiConnected = true;
          } else {
            !this.isApiConnected;
          }
        })
        .catch((error) => {
          console.error("Error checking connection status:", error);
          !this.isApiConnected;
        });
    },
    onUpload() {
      if (
        this.isApiConnected &&
        this.isFileSelected &&
        this.uploadStatus != "上傳中" &&
        this.currentFile != this.fileName
      ) {
        this.uploadStatus = "上傳中";
        this.isUploadSuccess = false;
        let formData = new FormData();
        formData.append("file", this.file);

        axios
          .post("http://localhost:8000/upload/", formData, {
            headers: {
              "Content-Type": "multipart/form-data",
            },
            responseType: "blob",
          })
          .then((res) => {
            const videoUrl = URL.createObjectURL(res.data);
            const videoPlayer = document.getElementById("videoPlayer");
            videoPlayer.src = videoUrl;
            this.currentFile = this.fileName;
            this.uploadStatus = "上傳成功";
          })
          .catch((err) => {
            console.log(err);
            this.uploadStatus = "上傳失敗";
          });
      }
    },
    onFileSelect(e) {
      const regex = /\.mp4$/i;
      this.file = e.target.files[0];
      if (regex.test(this.file.name)) {
        this.isFormatWrong = false;
        this.isFileSelected = true;
        this.fileName = this.file.name;
      } else {
        this.isFormatWrong = true;
        this.isFileSelected = false;
        this.fileName = "";
      }
    },
  },
};
</script>
