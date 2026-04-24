import onnxruntime
import numpy as np
import soundfile as sf
import librosa
import math


class HubertStreamingONNX:
    def __init__(self, modelpath):
        sess_opt = onnxruntime.SessionOptions()
        sess_opt.intra_op_num_threads = 16
        self.ort_session = onnxruntime.InferenceSession(
            modelpath,
            sess_options=sess_opt,
            providers=["CPUExecutionProvider"],
        )

    def forward(self, input_values):
        encoding_out = self.ort_session.run(
            None,
            {"input_values": input_values}
        )[0]
        return encoding_out
    

class W2F_HS_ONNX:
    def __init__(self, modelpath):
        self.encoder = HubertStreamingONNX(modelpath)

    def forward_chunk(self, waveform_chunk):
        # [chunk_size + 80]
        encoding_chunk = self.encoder.forward(
            waveform_chunk[None, :].astype(np.float32)
        )  # [chunk_size / 16000 / 0.02, 1024]

        return encoding_chunk
    

class Wav2FeatHubert:
    def __init__(self, modelpath):
        self.w2f = W2F_HS_ONNX(modelpath)

    def wav2feat(self, wav, npy="", chunksize=(3, 5, 2)):
        wav_input_16khz, _ = librosa.load(wav, sr=16000)
        speech = wav_input_16khz

        num_f = math.ceil(len(speech) / 16000 * 25)

        split_len = int(sum(chunksize) * 0.04 * 16000) + 80
        speech_pad = np.concatenate([
            np.zeros((split_len - int(sum(chunksize[1:]) * 0.04 * 16000),), dtype=speech.dtype),
            speech,
            np.zeros((split_len,), dtype=speech.dtype),
        ], 0)

        valid_feat_s = - sum(chunksize[1:]) * 2   # -7
        valid_feat_e = - chunksize[2] * 2   # -2
        i = 0
        res_lst = []
        while i < num_f:
            sss = int(i * 0.04 * 16000)
            eee = sss + split_len

            encoding_chunk = self.w2f.forward_chunk(speech_pad[sss:eee])   # (20, 1024)
            valid_encoding = encoding_chunk[valid_feat_s:valid_feat_e]
            # valid_feat = valid_encoding.reshape(chunksize[1], 2, 1024)
            valid_feat = valid_encoding.reshape(chunksize[1], 2, 1024).mean(1)    # [5, 1024]

            # valid_feat = valid_feat.cpu().numpy()
            res_lst.append(valid_feat)
            i += chunksize[1]
        
        ret = np.concatenate(res_lst, 0)
        ret = ret[:num_f]
        if npy:
            np.save(npy, ret)
        return ret

    def __call__(self, *args, **kwargs):
        return self.wav2feat(*args, **kwargs)


def get_hubert_stream_from_16k_wav(wav_16k, w2f: W2F_HS_ONNX, chunksize=[3, 5, 2]):
    speech_16k, sr = sf.read(wav_16k)
    assert sr == 16000, f"wav sr error: {sr}, need 16000"
    speech = speech_16k

    if speech.ndim == 2:
        speech = speech[:, 0]  # [T, 2] ==> [T,]
    
    num_f = int(len(speech) / 16000 * 25)

    split_len = int(sum(chunksize) * 0.04 * 16000) + 80
    speech_pad = np.concatenate([
        np.zeros((split_len - int(sum(chunksize[1:]) * 0.04 * 16000),), dtype=speech.dtype),
        speech,
        np.zeros((split_len,), dtype=speech.dtype),
    ], 0)

    valid_feat_s = - sum(chunksize[1:]) * 2   # -7
    valid_feat_e = - chunksize[2] * 2   # -2
    i = 0
    res_lst = []
    while i < num_f:
        sss = int(i * 0.04 * 16000)
        eee = sss + split_len

        encoding_chunk = w2f.forward_chunk(speech_pad[sss:eee])   # (20, 1024)
        valid_encoding = encoding_chunk[valid_feat_s:valid_feat_e]
        # valid_feat = valid_encoding.reshape(chunksize[1], 2, 1024)
        valid_feat = valid_encoding.reshape(chunksize[1], 2, 1024).mean(1)    # [5, 1024]
        # valid_feat = valid_feat.cpu().numpy()
        res_lst.append(valid_feat)
        i += chunksize[1]
    
    ret = np.concatenate(res_lst, 0)
    ret = ret[:num_f]
    return ret


if __name__ == '__main__':
    modelpath = "hubert_streaming_fix_kv.onnx"
    chunksize=[3, 5, 2]     # 长度可变

    w2f = W2F_HS_ONNX(modelpath)

    wav_16k = "test.wav"
    arr = get_hubert_stream_from_16k_wav(wav_16k, w2f, chunksize)
    print(arr.shape)

