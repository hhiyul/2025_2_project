package com.example.andro

import android.content.Context
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.andro.network.InferenceResponse
import com.example.andro.network.RetrofitInstance
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

class InferenceViewModel : ViewModel() {

    var uiState: InferenceUiState by mutableStateOf<InferenceUiState>(InferenceUiState.Idle)
        private set

    fun reset() {
        uiState = InferenceUiState.Idle
    }

    /** 서버 연결 테스트용 */
    fun checkHealth() {
        uiState = InferenceUiState.Loading

        viewModelScope.launch {
            try {
                val res = RetrofitInstance.api.healthCheck()
                if (res.isSuccessful) {
                    val status = res.body()?.get("status") ?: "unknown"
                    uiState = InferenceUiState.Success(
                        InferenceResponse(
                            filename = "health",
                            prediction = status,
                            confidence = 1.0f
                        )
                    )
                } else {
                    uiState = InferenceUiState.Error("health 실패: ${res.code()}")
                }
            } catch (e: Exception) {
                uiState = InferenceUiState.Error("health 에러: ${e.localizedMessage}")
            }
        }
    }

    /** 실제 이미지 추론 */
    fun inferImage(context: Context, uri: Uri) {
        uiState = InferenceUiState.Loading

        viewModelScope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
                        ?: throw IllegalArgumentException("이미지 읽기 실패")

                    val body = bytes.toRequestBody("image/jpeg".toMediaTypeOrNull())
                    val part = MultipartBody.Part.createFormData(
                        name = "file",           // FastAPI의 파라미터 이름과 같아야 함
                        filename = "image.jpg",
                        body = body
                    )

                    RetrofitInstance.api.infer(part)
                }

                if (response.isSuccessful && response.body() != null) {
                    uiState = InferenceUiState.Success(response.body()!!)
                } else {
                    uiState = InferenceUiState.Error("infer 실패: ${response.code()}")
                }
            } catch (e: Exception) {
                uiState = InferenceUiState.Error("infer 에러: ${e.localizedMessage}")
            }
        }
    }
}