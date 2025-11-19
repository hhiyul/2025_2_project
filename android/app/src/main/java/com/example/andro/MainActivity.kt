package com.example.andro

import android.os.Bundle
import android.Manifest
import android.content.Context
import android.net.Uri
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.rememberCoroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import android.content.pm.PackageManager
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import com.example.andro.network.InferenceResponse
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.tooling.preview.PreviewScreenSizes
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import coil.compose.AsyncImage
import com.example.andro.ui.theme.AndroTheme
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import android.webkit.MimeTypeMap
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import org.json.JSONObject


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AndroTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    CameraAndGalleryScreen()
                }
            }
        }
    }
}

@PreviewScreenSizes
@Composable
fun CameraAndGalleryScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    var pendingCameraUri by remember { mutableStateOf<Uri?>(null) }
    var uiState by remember { mutableStateOf<InferenceUiState>(InferenceUiState.Idle) }
    val scope = rememberCoroutineScope()
    val takePictureLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            selectedImageUri = pendingCameraUri
            uiState = InferenceUiState.Idle
        } else {
            uiState = InferenceUiState.Error("촬영이 취소되었거나 실패했습니다.")
            pendingCameraUri?.let { uri ->
                runCatching { context.contentResolver.delete(uri, null, null) }
            }
        }
    }

    fun launchCamera() {
        val uri = runCatching { createImageUri(context) }
            .onFailure { throwable ->
                uiState = InferenceUiState.Error(throwable.localizedMessage ?: "카메라를 실행할 수 없습니다.")
            }
            .getOrNull()

        if (uri != null) {
            pendingCameraUri = uri
            uiState = InferenceUiState.Idle
            takePictureLauncher.launch(uri)
        }
    }

    // 카메라 권한 런처
    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            launchCamera()
        } else {
            uiState = InferenceUiState.Error("카메라 권한이 필요합니다.")
        }
    }

    // 앨범 선택 런처
    val pickVisualMediaLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        selectedImageUri = uri
        uiState = InferenceUiState.Idle
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "사진을 촬영하거나 앨범에서 이미지를 선택하세요.",
            style = MaterialTheme.typography.titleMedium,
            textAlign = TextAlign.Center
        )

        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f, fill = true),
            tonalElevation = 2.dp,
            shape = MaterialTheme.shapes.medium
        ) {
            if (selectedImageUri != null) {
                AsyncImage(
                    model = selectedImageUri,
                    contentDescription = "선택된 이미지",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
            } else {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "이미지가 여기에 표시됩니다.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(24.dp)
                    )
                }
            }
        }

        // 추론 상태 안내
        when (val state = uiState) {
            is InferenceUiState.Error -> {
                Text(
                    text = state.message,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth()
                )
            }

            is InferenceUiState.Success -> {
                Text(
                    text = "예측 결과: ${state.response.prediction}",
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth()
                )
            }

            InferenceUiState.Idle -> {
                Text(
                    text = "이미지를 확인한 뒤 추론하기를 눌러주세요.",
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            InferenceUiState.Loading -> {
                Text(
                    text = "잠시만 기달려 주세요...",
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }

        // 추론하기
        Button(
            enabled = uiState !is InferenceUiState.Loading,
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp)
                .padding(horizontal = 16.dp),
            onClick = {
                val uri = selectedImageUri
                if (uri == null) {
                    uiState = InferenceUiState.Error("이미지를 먼저 선택하거나 촬영하세요.")
                    return@Button
                }

                scope.launch {
                    uiState = InferenceUiState.Loading
                    try {
                        val result = uploadAndInfer(context, uri)
                        uiState = InferenceUiState.Success(result)
                    } catch (e: Exception) {
                        uiState = InferenceUiState.Error(
                            e.localizedMessage ?: "추론 요청 중 오류가 발생했습니다."
                        )
                    }
                }
            }
        ) {
            Text(
                if (uiState is InferenceUiState.Loading) "추론 중..."
                else "추론하기"
            )
        }

        // 하단 버튼들
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Button(
                modifier = Modifier
                    .weight(1f)
                    .height(40.dp),
                onClick = {
                    if (
                        ContextCompat.checkSelfPermission(
                            context,
                            Manifest.permission.CAMERA
                        ) == PackageManager.PERMISSION_GRANTED
                    ) {
                        launchCamera()
                    } else {
                        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                }
            ) {
                Text("카메라로 촬영")
            }

            Button(
                modifier = Modifier
                    .weight(1f)
                    .height(40.dp),
                onClick = {
                    pickVisualMediaLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                }
            ) {
                Text("앨범에서 선택")
            }
        }
        Spacer(modifier = Modifier.height(8.dp))
    }
}

private fun createImageUri(context: Context): Uri? {
    val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
    val imageDir = File(context.cacheDir, "camera").apply {
        if (!exists()) mkdirs()
    }
    val imageFile = File.createTempFile("IMG_$timeStamp", ".jpg", imageDir)
    return FileProvider.getUriForFile(
        context,
        "${context.packageName}.fileprovider",
        imageFile
    )
}

// ===== Retrofit 인터페이스 & 데이터 클래스 =====
interface InferenceApi {
    @Multipart
    @POST("infer")
    suspend fun infer(@Part file: MultipartBody.Part): InferenceResponse
}


// ===== 업로드 + 추론 호출 =====
suspend fun uploadAndInfer(context: Context, uri: Uri): InferenceResponse =
    withContext(Dispatchers.IO) {
        // 에뮬레이터 → 로컬 FastAPI 서버
        // 실제 기기 + ngrok 쓰면 baseUrl을 ngrok URL로 바꾸면 됨
        val client = OkHttpClient.Builder()
            .addInterceptor { chain ->
                val newReq = chain.request().newBuilder()
                    .addHeader("X-API-Key", "fuck-key-123") // 🔑 FastAPI 인증 헤더
                    .build()
                chain.proceed(newReq)
            }
            .build()

        val retrofit = Retrofit.Builder()
            .baseUrl("https://uncially-engrossing-keeley.ngrok-free.dev/")   // PC에서 FastAPI가 돌고 있을 때
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        val api = retrofit.create(InferenceApi::class.java)

        val cr = context.contentResolver
        val mime = cr.getType(uri) ?: run {
            val ext = MimeTypeMap.getFileExtensionFromUrl(uri.toString())
            MimeTypeMap.getSingleton().getMimeTypeFromExtension(ext) ?: "application/octet-stream"
        }

        val name = runCatching {
            cr.query(uri, null, null, null, null)?.use { c ->
                val nameIdx = c.getColumnIndex("_display_name")
                if (c.moveToFirst() && nameIdx >= 0) c.getString(nameIdx) else null
            }
        }.getOrNull() ?: "upload." + (MimeTypeMap.getSingleton().getExtensionFromMimeType(mime) ?: "jpg")

        val bytes = cr.openInputStream(uri)?.use { it.readBytes() }
            ?: error("이미지 열기 실패")

        val body = bytes.toRequestBody(mime.toMediaTypeOrNull())
        val part = MultipartBody.Part.createFormData("file", name, body)

        api.infer(part)
    }

fun formatInferenceResultAsJson(result: InferenceResponse): String {
    val json = JSONObject().apply {
        put("filename", result.filename ?: JSONObject.NULL)
        put("content_type", result.content_type ?: JSONObject.NULL)
        put("size_bytes", result.size_bytes)
        put("prediction", result.prediction ?: JSONObject.NULL)
        put("confidence", result.confidence)
    }
    return json.toString(2)
}

@Composable
fun ResultScreen(prediction: String) {
    Text("예측 결과: $prediction")
}