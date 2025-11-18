package com.example.andro

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.compose.runtime.rememberCoroutineScope
import androidx.activity.compose.rememberLauncherForActivityResult
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.ui.tooling.preview.PreviewScreenSizes
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import coil.compose.AsyncImage
import com.example.andro.network.InferenceResponse
import com.example.andro.network.RetrofitInstance
import com.example.andro.ui.theme.AndroTheme


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

@Composable
fun CameraAndGalleryScreen(
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    var pendingCameraUri by remember { mutableStateOf<Uri?>(null) }

    var uiState by remember { mutableStateOf<InferenceUiState>(InferenceUiState.Idle) }
    var localMessage by remember { mutableStateOf<String?>(null) }

    // 사진 촬영 런처
    val takePictureLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            selectedImageUri = pendingCameraUri
            localMessage = null
            uiState = InferenceUiState.Idle
        } else {
            localMessage = "촬영이 취소되었거나 실패했습니다."
            pendingCameraUri?.let { uri ->
                runCatching { context.contentResolver.delete(uri, null, null) }
            }
        }
    }

    fun launchCamera() {
        val uri = runCatching { createImageUri(context) }
            .onFailure { throwable ->
                localMessage = throwable.localizedMessage ?: "카메라를 실행할 수 없습니다."
            }
            .getOrNull()

        if (uri != null) {
            pendingCameraUri = uri
            localMessage = null
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
            localMessage = "카메라 권한이 필요합니다."
        }
    }

    // 앨범 선택 런처
    val pickVisualMediaLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        selectedImageUri = uri
        localMessage = null
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

        // 이미지 미리보기
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

        // 상태 메시지
        val messageFromState = when (val state = uiState) {
            is InferenceUiState.Error   -> state.message
            is InferenceUiState.Success -> formatInferenceResultAsJson(state.response)
            InferenceUiState.Idle       -> "선택한 이미지를 확인한 뒤 추론을 실행하세요."
            InferenceUiState.Loading    -> "추론을 실행 중입니다..."
        }

        val isError = localMessage != null || uiState is InferenceUiState.Error
        val textColor = if (isError) {
            MaterialTheme.colorScheme.error
        } else {
            MaterialTheme.colorScheme.onSurface
        }

        Text(
            text = localMessage ?: messageFromState,
            color = textColor,
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )

        // 🔥 추론 버튼
        Button(
            enabled = uiState !is InferenceUiState.Loading,
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp)
                .padding(horizontal = 16.dp),
            onClick = {
                val uri = selectedImageUri
                if (uri == null) {
                    localMessage = "이미지를 먼저 선택하거나 촬영하세요."
                    return@Button
                }

                localMessage = null
                uiState = InferenceUiState.Loading

                scope.launch {
                    try {
                        // 이미지 → 바이트 배열
                        val bytes = withContext(Dispatchers.IO) {
                            context.contentResolver
                                .openInputStream(uri)
                                ?.use { it.readBytes() }
                                ?: throw IllegalArgumentException("이미지 읽기 실패")
                        }

                        // Multipart 파일 생성
                        val body = bytes.toRequestBody("image/jpeg".toMediaTypeOrNull())
                        val part = MultipartBody.Part.createFormData(
                            name = "file",
                            filename = "image.jpg",
                            body = body
                        )

                        // FastAPI /infer 호출
                        val response = withContext(Dispatchers.IO) {
                            RetrofitInstance.api.infer(part)
                        }

                        if (response.isSuccessful && response.body() != null) {
                            uiState = InferenceUiState.Success(response.body()!!)
                        } else {
                            uiState = InferenceUiState.Error(
                                "infer 실패: ${response.code()} ${response.message()}"
                            )
                        }
                    } catch (e: Exception) {
                        uiState = InferenceUiState.Error(
                            "infer 에러: ${e.localizedMessage ?: "알 수 없는 오류"}"
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

        // 하단 버튼: 카메라 / 앨범
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

// ----- 이미지 파일 URI 생성 -----
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

// ----- 추론 결과 JSON 포맷 -----
fun formatInferenceResultAsJson(result: InferenceResponse): String {
    val json = JSONObject().apply {
        put("filename", result.filename)
        put("prediction", result.prediction)
        put("confidence", result.confidence)
    }
    return json.toString(2)
}
@PreviewScreenSizes
@Composable
private fun CameraAndGalleryScreenPreview() {
    AndroTheme {
        CameraAndGalleryScreen()
    }
}