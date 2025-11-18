package com.example.andro

import com.example.andro.network.InferenceResponse

sealed interface InferenceUiState {
    data object Idle : InferenceUiState
    data object Loading : InferenceUiState
    data class Success(val response: InferenceResponse) : InferenceUiState
    data class Error(val message: String) : InferenceUiState
}