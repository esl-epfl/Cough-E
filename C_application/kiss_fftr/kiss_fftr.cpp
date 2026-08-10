/*
Copyright (c) 2003-2004, Mark Borgerding

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the author nor the names of any contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <helpers.h>

#include "twiddles_win08_fs8000.h"
#include "kiss_fftr.h"
#include "_kiss_fft_guts.h"

/*
    Function to read the precomputed super_twiddle factors inside the FFTR cfg structure
*/
template<typename T>
void init_super_twiddles(kiss_fftr_cfg<T> *st, const twiddles_t<T> *twiddles, int16_t len);

#include "string.h"
void tostring(char str[], int num)
{
    int i, rem, len = 0, n;

    n = num;
    while (n != 0)
    {
        len++;
        n /= 10;
    }
    for (i = 0; i < len; i++)
    {
        rem = num % 10;
        num = num / 10;
        str[len - (i + 1)] = rem + '0';
    }
    str[len] = '\0';
}

template<typename T>
struct kiss_fftr_state {
    kiss_fft_cfg<T> substate;
    kiss_fft_cpx<T> * tmpbuf;
    kiss_fft_cpx<T> * super_twiddles;
#ifdef USE_SIMD
    void * pad;
#endif
};

template<typename T>
kiss_fftr_cfg<T> kiss_fftr_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem)
{
    kiss_fftr_cfg<T> st = NULL;
    size_t subsize, memneeded;

    if (nfft & 1) {
        printf("Real FFT optimization must be even.\n");
        return NULL;
    }
    nfft >>= 1;

    // printf("-call fft alloc with nfft=%d\n", nfft);
    kiss_fft_alloc<T> (nfft, inverse_fft, NULL, &subsize);
    memneeded = sizeof(struct kiss_fftr_state<T>) + subsize + sizeof(kiss_fft_cpx<T>) * ( nfft * 3 / 2);

    // printf("Size struct kiss_fftr_state: %zu\n", sizeof(struct kiss_fftr_state<T>));
    // printf("%zu\n%zu\n%zu\n%d\n", sizeof(struct kiss_fftr_state<T>), subsize, sizeof(kiss_fft_cpx<T>), nfft);
    // printf("%zu\n\n", memneeded);
    // exit(0);

    if (lenmem == NULL) {
        // printf("lenmem = NULL\n");
        st = (kiss_fftr_cfg<T>) KISS_FFT_MALLOC (memneeded);
    } else {
        // printf("else lenmeme!=NULL\n");
        if (*lenmem >= memneeded){
            // printf("lenmeme >= memneeded\n");
            st = (kiss_fftr_cfg<T>) mem;
        }
        *lenmem = memneeded;
    }
    if (!st){
        // printf("return NULL\n");
        return NULL;
    }

    st->substate = (kiss_fft_cfg<T>) (st + 1); /*just beyond kiss_fftr_state struct */
    st->tmpbuf = (kiss_fft_cpx<T> *) (((char *) st->substate) + subsize);
    st->super_twiddles = st->tmpbuf + nfft;
    kiss_fft_alloc<T>(nfft, inverse_fft, st->substate, &subsize);

    // // Online computation of the twiddles factors
    // printf("\nFFTR: %d\n", nfft/2);
    // for (int i = 0; i < nfft/2; ++i) {
    //     double phase =
    //         -3.14159265358979323846264338327 * ((double) (i+1) / nfft + .5);
    //     if (inverse_fft)
    //         phase *= -1;
    //     kf_cexp (st->super_twiddles+i,phase);

    //     printf("%f\t\t%f\n", cos(phase), sin(phase));
    // }


    ///    FINAL AUDIO MODEL ///
    /* Differentiate between the cases (application specific) */
    switch (nfft/2)
    {
    case 225:
        init_super_twiddles(&st, twiddles_225<T>, nfft/2);
        break;

    case 512:
        init_super_twiddles(&st, twiddles_512<T>, nfft/2);
        break;

    case 1600:
        init_super_twiddles(&st, twiddles_1600<T>, nfft/2);
        break;

    default:
        printf("NFFT initialization error!\n");
        return NULL;
        break;
    }

    // ///    UNOPTIMIZED AUDIO MODEL ///
    // /* Differentiate between the cases (application specific) */
    // switch (nfft/2)
    // {
    // case 4000:
    //     init_super_twiddles(&st, twiddles_4000<T>, nfft/2);
    //     break;

    // case 512:
    //     init_super_twiddles(&st, twiddles_512<T>, nfft/2);
    //     break;

    // case 225:
    //     init_super_twiddles(&st, twiddles_225<T>, nfft/2);
    //     break;

    // default:
    //     printf("NFFT initialization error!\n");
    //     return NULL;
    //     break;
    // }

    return st;
}


/*
    Function to read the precomputed super_twiddle factors inside the FFTR cfg structure
*/
template<typename T>
void init_super_twiddles(kiss_fftr_cfg<T> *st, const twiddles_t<T> *twiddles, int16_t len){
    // we assume (*st)->twiddles[i] is defined with r-i order
    // and twiddles[i] is defined as cosine-sine order
    read_flash(twiddles, (*st)->super_twiddles, len*sizeof(twiddles_t<T>));
}

template<typename T>
void kiss_fftr(kiss_fftr_cfg<T> st,const T *timedata,kiss_fft_cpx<T> *freqdata)
{
    /* input buffer timedata is stored row-wise */
    int k,ncfft;
    kiss_fft_cpx<T> fpnk,fpk,f1k,f2k,tw,tdc;

    if ( st->substate->inverse) {
        printf("kiss fft usage error: improper alloc\n");
        exit(1);
    }

    ncfft = st->substate->nfft;

    /*perform the parallel fft of two real signals packed in real,imag*/
    kiss_fft( st->substate , (const kiss_fft_cpx<T>*)timedata, st->tmpbuf );
    /* The real part of the DC element of the frequency spectrum in st->tmpbuf
     * contains the sum of the even-numbered elements of the input time sequence
     * The imag part is the sum of the odd-numbered elements
     *
     * The sum of tdc.r and tdc.i is the sum of the input time sequence.
     *      yielding DC of input time sequence
     * The difference of tdc.r - tdc.i is the sum of the input (dot product) [1,-1,1,-1...
     *      yielding Nyquist bin of input time sequence
     */

    tdc.r = st->tmpbuf[0].r;
    tdc.i = st->tmpbuf[0].i;
    C_FIXDIV(tdc,2);
    CHECK_OVERFLOW_OP(tdc.r ,+, tdc.i);
    CHECK_OVERFLOW_OP(tdc.r ,-, tdc.i);
    freqdata[0].r = tdc.r + tdc.i;
    freqdata[ncfft].r = tdc.r - tdc.i;
#ifdef USE_SIMD
    freqdata[ncfft].i = freqdata[0].i = _mm_set1_ps(0);
#else
    freqdata[ncfft].i = freqdata[0].i = CONST_ZERO;
#endif

    for ( k=1;k <= ncfft/2 ; ++k ) {
        fpk    = st->tmpbuf[k];
        fpnk.r =   st->tmpbuf[ncfft-k].r;
        fpnk.i = - st->tmpbuf[ncfft-k].i;
        C_FIXDIV(fpk,2);
        C_FIXDIV(fpnk,2);

        C_ADD( f1k, fpk , fpnk );
        C_SUB( f2k, fpk , fpnk );
        C_MUL( tw , f2k , st->super_twiddles[k-1]);

        freqdata[k].r = HALF_OF(f1k.r + tw.r);
        freqdata[k].i = HALF_OF(f1k.i + tw.i);
        freqdata[ncfft-k].r = HALF_OF(f1k.r - tw.r);
        freqdata[ncfft-k].i = HALF_OF(tw.i - f1k.i);
    }
}

template<typename T>
void kiss_fftri(kiss_fftr_cfg<T> st,const kiss_fft_cpx<T> *freqdata,T *timedata)
{
    /* input buffer timedata is stored row-wise */
    int k, ncfft;

    if (st->substate->inverse == 0) {
        printf("kiss fft usage error: improper alloc\n");
        exit (1);
    }

    ncfft = st->substate->nfft;

    st->tmpbuf[0].r = freqdata[0].r + freqdata[ncfft].r;
    st->tmpbuf[0].i = freqdata[0].r - freqdata[ncfft].r;
    C_FIXDIV(st->tmpbuf[0],2);

    for (k = 1; k <= ncfft / 2; ++k) {
        kiss_fft_cpx<T> fk, fnkc, fek, fok, tmp;
        fk = freqdata[k];
        fnkc.r = freqdata[ncfft - k].r;
        fnkc.i = -freqdata[ncfft - k].i;
        C_FIXDIV( fk , 2 );
        C_FIXDIV( fnkc , 2 );

        C_ADD (fek, fk, fnkc);
        C_SUB (tmp, fk, fnkc);
        C_MUL (fok, tmp, st->super_twiddles[k-1]);
        C_ADD (st->tmpbuf[k],     fek, fok);
        C_SUB (st->tmpbuf[ncfft - k], fek, fok);
#ifdef USE_SIMD
        st->tmpbuf[ncfft - k].i *= _mm_set1_ps(-1.0);
#else
        st->tmpbuf[ncfft - k].i *= CONST_NEG_ONE;
#endif
    }
    kiss_fft (st->substate, st->tmpbuf, (kiss_fft_cpx<T> *) timedata);
}

#ifdef FXP_MODE
template struct kiss_fftr_state<kiss_fft_scalar>;
template kiss_fftr_cfg<kiss_fft_scalar> kiss_fftr_alloc<kiss_fft_scalar>(int nfft, int inverse_fft, void * mem, size_t * lenmem);
template void kiss_fftr<kiss_fft_scalar>(kiss_fftr_cfg<kiss_fft_scalar> cfg, const kiss_fft_scalar *timedata, kiss_fft_cpx<kiss_fft_scalar> *freqdata);
template void kiss_fftri<kiss_fft_scalar>(kiss_fftr_cfg<kiss_fft_scalar> cfg, const kiss_fft_cpx<kiss_fft_scalar> *freqdata, kiss_fft_scalar *timedata);
#else
template struct kiss_fftr_state<sreal_t>;
template kiss_fftr_cfg<sreal_t> kiss_fftr_alloc<sreal_t>(int nfft, int inverse_fft, void * mem, size_t * lenmem);
template void kiss_fftr<sreal_t>(kiss_fftr_cfg<sreal_t> cfg, const sreal_t *timedata, kiss_fft_cpx<sreal_t> *freqdata);
template void kiss_fftri<sreal_t>(kiss_fftr_cfg<sreal_t> cfg, const kiss_fft_cpx<sreal_t> *freqdata, sreal_t *timedata);
#endif
