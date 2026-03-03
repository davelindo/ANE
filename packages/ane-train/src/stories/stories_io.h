// stories_io.h — IOSurface helpers, blob builders, NEON conversion
#pragma once
#include "stories_config.h"
#include <arm_neon.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>

static IOSurfaceRef make_surface(size_t bytes) {
  return IOSurfaceCreate((__bridge CFDictionaryRef) @{
    (id)kIOSurfaceWidth : @(bytes),
    (id)kIOSurfaceHeight : @1,
    (id)kIOSurfaceBytesPerElement : @1,
    (id)kIOSurfaceBytesPerRow : @(bytes),
    (id)kIOSurfaceAllocSize : @(bytes),
    (id)kIOSurfacePixelFormat : @0
  });
}

static NSData *build_blob(const float *w, int rows, int cols) {
  int ws = rows * cols * 2, tot = 128 + ws;
  uint8_t *b = (uint8_t *)calloc(tot, 1);
  b[0] = 1;
  b[4] = 2;
  b[64] = 0xEF;
  b[65] = 0xBE;
  b[66] = 0xAD;
  b[67] = 0xDE;
  b[68] = 1;
  *(uint32_t *)(b + 72) = ws;
  *(uint32_t *)(b + 80) = 128;
  _Float16 *fp16 = (_Float16 *)(b + 128);
  for (int i = 0; i < rows * cols; i++)
    fp16[i] = (_Float16)w[i];
  return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}
static NSData *build_blob_t(const float *w, int rows, int cols) {
  int ws = cols * rows * 2, tot = 128 + ws;
  uint8_t *b = (uint8_t *)calloc(tot, 1);
  b[0] = 1;
  b[4] = 2;
  b[64] = 0xEF;
  b[65] = 0xBE;
  b[66] = 0xAD;
  b[67] = 0xDE;
  b[68] = 1;
  *(uint32_t *)(b + 72) = ws;
  *(uint32_t *)(b + 80) = 128;
  _Float16 *fp16 = (_Float16 *)(b + 128);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      fp16[j * rows + i] = (_Float16)w[i * cols + j];
  return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}
static NSData *build_blob_fp16(_Float16 *d, int cnt) {
  int ws = cnt * 2, tot = 128 + ws;
  uint8_t *b = (uint8_t *)calloc(tot, 1);
  b[0] = 1;
  b[4] = 2;
  b[64] = 0xEF;
  b[65] = 0xBE;
  b[66] = 0xAD;
  b[67] = 0xDE;
  b[68] = 1;
  *(uint32_t *)(b + 72) = ws;
  *(uint32_t *)(b + 80) = 128;
  memcpy(b + 128, d, ws);
  return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// NEON vectorized conversion
static void cvt_f16_f32(float *dst, const _Float16 *src, int n) {
  int i = 0;
  for (; i + 7 < n; i += 8) {
    float16x8_t h = vld1q_f16((const __fp16 *)(src + i));
    vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(h)));
    vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(h)));
  }
  for (; i < n; i++)
    dst[i] = (float)src[i];
}
static void cvt_f32_f16(_Float16 *dst, const float *src, int n) {
  int i = 0;
  for (; i + 7 < n; i += 8) {
    float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src + i)),
                                 vcvt_f16_f32(vld1q_f32(src + i + 4)));
    vst1q_f16((__fp16 *)(dst + i), h);
  }
  for (; i < n; i++)
    dst[i] = (_Float16)src[i];
}

// IOSurface I/O (channel-first [C,S] layout)
static void io_write_fp16(IOSurfaceRef s, const float *data, int channels,
                          int sp) {
  IOSurfaceLock(s, 0, NULL);
  cvt_f32_f16((_Float16 *)IOSurfaceGetBaseAddress(s), data, channels * sp);
  IOSurfaceUnlock(s, 0, NULL);
}
static void io_read_fp16(IOSurfaceRef s, float *data, int ch_off, int channels,
                         int sp) {
  IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
  cvt_f16_f32(data, (_Float16 *)IOSurfaceGetBaseAddress(s) + ch_off * sp,
              channels * sp);
  IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}
static void io_copy(IOSurfaceRef dst, int dst_ch, IOSurfaceRef src, int src_ch,
                    int channels, int sp) {
  IOSurfaceLock(dst, 0, NULL);
  IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL);
  memcpy((_Float16 *)IOSurfaceGetBaseAddress(dst) + dst_ch * sp,
         (_Float16 *)IOSurfaceGetBaseAddress(src) + src_ch * sp,
         channels * sp * sizeof(_Float16));
  IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
  IOSurfaceUnlock(dst, 0, NULL);
}
static void io_write_fp16_at(IOSurfaceRef s, int ch_off, const float *data,
                             int channels, int sp) {
  IOSurfaceLock(s, 0, NULL);
  cvt_f32_f16((_Float16 *)IOSurfaceGetBaseAddress(s) + ch_off * sp, data,
              channels * sp);
  IOSurfaceUnlock(s, 0, NULL);
}

// Kernel compile/eval + runtime tuning
static int g_runtime_cfg_inited = 0;
static int g_runtime_queue_depth = 2;
static int g_runtime_use_client_eval = 1;
static int g_runtime_perf_stats = 1;
static int g_runtime_alias_ffn_io = 1;
static int g_runtime_perf_array_arg = 1;
static unsigned int g_runtime_perf_mask = 0xFFFFFFFFu;

static int env_int_or_default(const char *name, int dflt) {
  const char *s = getenv(name);
  if (!s || !*s)
    return dflt;
  if (s[0] == 't' || s[0] == 'T' || s[0] == 'y' || s[0] == 'Y')
    return 1;
  if (s[0] == 'f' || s[0] == 'F' || s[0] == 'n' || s[0] == 'N')
    return 0;
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (end == s)
    return dflt;
  return (int)v;
}

static unsigned int env_u32_or_default(const char *name, unsigned int dflt) {
  const char *s = getenv(name);
  if (!s || !*s)
    return dflt;
  errno = 0;
  char *end = NULL;
  unsigned long v = strtoul(s, &end, 0);
  if (errno != 0 || end == s || v > UINT_MAX)
    return dflt;
  return (unsigned int)v;
}

static void ane_runtime_cfg_init(void) {
  if (g_runtime_cfg_inited)
    return;
  g_runtime_cfg_inited = 1;
  g_runtime_queue_depth = env_int_or_default("ANE_QUEUE_DEPTH", 2);
  if (g_runtime_queue_depth < 0)
    g_runtime_queue_depth = 0;
  if (g_runtime_queue_depth > 63)
    g_runtime_queue_depth = 63;
  g_runtime_use_client_eval = env_int_or_default("ANE_USE_CLIENT_EVAL", 1);
  g_runtime_perf_stats = env_int_or_default("ANE_PERF_STATS", 1);
  g_runtime_alias_ffn_io = env_int_or_default("ANE_ALIAS_FFN_IO", 1);
  g_runtime_perf_array_arg = env_int_or_default("ANE_PERF_ARRAY_ARG", 1);
  g_runtime_perf_mask = env_u32_or_default("ANE_PERF_MASK", 0xFFFFFFFFu);
}

static int ane_cfg_queue_depth(void) {
  ane_runtime_cfg_init();
  return g_runtime_queue_depth;
}
static int ane_cfg_use_client_eval(void) {
  ane_runtime_cfg_init();
  return g_runtime_use_client_eval;
}
static int ane_cfg_perf_stats(void) {
  ane_runtime_cfg_init();
  return g_runtime_perf_stats;
}
static int ane_cfg_alias_ffn_io(void) {
  ane_runtime_cfg_init();
  return g_runtime_alias_ffn_io;
}

static void ane_apply_model_tuning(id mdl) {
  if (!mdl)
    return;
  ane_runtime_cfg_init();
  if (g_runtime_queue_depth > 0 &&
      [mdl respondsToSelector:@selector(setQueueDepth:)]) {
    ((void (*)(id, SEL, signed char))objc_msgSend)(
        mdl, @selector(setQueueDepth:), (signed char)g_runtime_queue_depth);
  }
  if (g_runtime_perf_stats &&
      [mdl respondsToSelector:@selector(setPerfStatsMask:)]) {
    unsigned int mask = g_runtime_perf_mask;
    if (g_APS && [g_APS respondsToSelector:@selector(driverMaskForANEFMask:)]) {
      mask = ((unsigned int (*)(id, SEL, unsigned int))objc_msgSend)(
          g_APS, @selector(driverMaskForANEFMask:), mask);
    }
    ((void (*)(id, SEL, unsigned int))objc_msgSend)(
        mdl, @selector(setPerfStatsMask:), mask);
  }
}

static id ane_make_perf_stats(void) {
  ane_runtime_cfg_init();
  if (!g_runtime_perf_stats || !g_APS)
    return nil;
  id perf = nil;
  if ([g_APS respondsToSelector:@selector(statsWithHardwareExecutionNS:)]) {
    perf = ((id (*)(id, SEL, uint64_t))objc_msgSend)(
        g_APS, @selector(statsWithHardwareExecutionNS:), (uint64_t)0);
  }
  if (!perf) {
    perf = ((id (*)(id, SEL))objc_msgSend)([g_APS alloc], @selector(init));
  }
  return perf;
}

static id ane_make_request(IOSurfaceRef ioIn, IOSurfaceRef ioOut, id perfStats,
                           NSNumber *procIndex) {
  id wI = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
      g_AIO, @selector(objectWithIOSurface:), ioIn);
  id wO = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
      g_AIO, @selector(objectWithIOSurface:), ioOut);
  NSNumber *pidx = procIndex ? procIndex : @0;
  id perfArg = perfStats;
  if (perfStats && g_runtime_perf_array_arg)
    perfArg = @[ perfStats ];
  return ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
      g_AR,
      @selector(requestWithInputs:
                     inputIndices:outputs:outputIndices:weightsBuffer:perfStats
                                 :procedureIndex:),
      @[ wI ], @[ @0 ], @[ wO ], @[ @0 ], nil, perfArg, pidx);
}

static bool ane_request_validate(id req) {
  if (!req || ![req respondsToSelector:@selector(validate)])
    return true;
  @try {
    return ((BOOL (*)(id, SEL))objc_msgSend)(req, @selector(validate));
  } @catch (NSException *ex) {
    (void)ex;
    return false;
  }
}

static bool kern_rebuild_request(Kern *k) {
  if (!k || !k->ioIn || !k->ioOut)
    return false;
  NSNumber *procIndex = @0;
  if (k->request) {
    id oldReq = (__bridge id)k->request;
    if ([oldReq respondsToSelector:@selector(procedureIndex)]) {
      id p = ((id (*)(id, SEL))objc_msgSend)(oldReq, @selector(procedureIndex));
      if ([p isKindOfClass:[NSNumber class]])
        procIndex = p;
    }
  }
  id perfStats = k->perfStats ? (__bridge id)k->perfStats : nil;
  id req = ane_make_request(k->ioIn, k->ioOut, perfStats, procIndex);
  if (!req)
    return false;
  if (k->request)
    CFRelease(k->request);
  k->request = (void *)CFBridgingRetain(req);
  return true;
}

static bool kern_alias_input_surface(Kern *dst, IOSurfaceRef sharedIn) {
  if (!dst || !sharedIn)
    return false;
  if (dst->ioIn == sharedIn)
    return true;
  if (dst->ioIn)
    CFRelease(dst->ioIn);
  dst->ioIn = (IOSurfaceRef)CFRetain(sharedIn);
  return kern_rebuild_request(dst);
}

static bool ane_alias_ffn_bwd_input(Kern *fwdFFN, Kern *ffnBwd) {
  if (!ane_cfg_alias_ffn_io())
    return true;
  if (!fwdFFN || !ffnBwd)
    return false;
  return kern_alias_input_surface(ffnBwd, fwdFFN->ioOut);
}

static double kern_last_hw_ms(Kern *k) {
  return (k && k->lastHwExecNS) ? ((double)k->lastHwExecNS / 1e6) : 0.0;
}

static void free_kern(Kern *k);

static Kern *compile_kern_mil_w(NSString *mil, NSDictionary *weights,
                                int ic_bytes, int oc_bytes) {
  @autoreleasepool {
    ane_runtime_cfg_init();
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
        g_D, @selector(modelWithMILText:weights:optionsPlist:), md, weights,
        nil);
    if (!desc) {
      printf("  [compile] desc=NULL\n");
      return NULL;
    }
    id mdl = ((id (*)(Class, SEL, id))objc_msgSend)(
        g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx =
        ((id (*)(id, SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager]
              createDirectoryAtPath:
                  [td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES
                         attributes:nil
                              error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"]
         atomically:YES];
    for (NSString *path in weights) {
      NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/"
                                                      withString:@""];
      [weights[path][@"data"]
          writeToFile:[td stringByAppendingPathComponent:rel]
           atomically:YES];
    }
    NSError *e = nil;
    if (!((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
      printf("  [compile] FAIL: %s\n",
             e ? [[e description] UTF8String] : "no error");
      return NULL;
    }
    if (!((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
      printf("  [compile] load FAIL\n");
      return NULL;
    }
    ane_apply_model_tuning(mdl);
    __sync_fetch_and_add(&g_compile_count, 1);
    Kern *k = (Kern *)calloc(1, sizeof(Kern));
    k->model = (void *)CFBridgingRetain(mdl);
    k->ioIn = make_surface(ic_bytes);
    k->ioOut = make_surface(oc_bytes);
    id perfStats = ane_make_perf_stats();
    if (perfStats)
      k->perfStats = (void *)CFBridgingRetain(perfStats);
    id req = ane_make_request(k->ioIn, k->ioOut, perfStats, @0);
    if (req && !ane_request_validate(req) && perfStats) {
      req = ane_make_request(k->ioIn, k->ioOut, nil, @0);
      if (k->perfStats) {
        CFRelease(k->perfStats);
        k->perfStats = NULL;
      }
      if (g_runtime_perf_stats) {
        fprintf(
            stderr,
            "  [compile] perfStats request rejected; disabling perfStats\n");
      }
      g_runtime_perf_stats = 0;
    }
    if (!req) {
      printf("  [compile] request creation failed\n");
      free_kern(k);
      return NULL;
    }
    k->request = (void *)CFBridgingRetain(req);
    if (g_runtime_use_client_eval &&
        [mdl respondsToSelector:@selector(sharedConnection)]) {
      id client =
          ((id (*)(id, SEL))objc_msgSend)(mdl, @selector(sharedConnection));
      if (client)
        k->client = (void *)CFBridgingRetain(client);
    }
    k->clientEvalEnabled = (k->client != NULL) && g_runtime_use_client_eval;
    k->tmpDir = (void *)CFBridgingRetain(td);
    return k;
  }
}
static void free_kern(Kern *k) {
  if (!k)
    return;
  if (k->model) {
    id mdl = (__bridge id)k->model;
    NSError *e = nil;
    ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
        mdl, @selector(unloadWithQoS:error:), 21, &e);
  }
  if (k->ioIn)
    CFRelease(k->ioIn);
  if (k->ioOut)
    CFRelease(k->ioOut);
  if (k->tmpDir) {
    [[NSFileManager defaultManager] removeItemAtPath:(__bridge id)k->tmpDir
                                               error:nil];
  }
  if (k->model)
    CFRelease(k->model);
  if (k->request)
    CFRelease(k->request);
  if (k->tmpDir)
    CFRelease(k->tmpDir);
  if (k->client)
    CFRelease(k->client);
  if (k->perfStats)
    CFRelease(k->perfStats);
  free(k);
}
static bool ane_eval(Kern *k) {
  if (!k || !k->model || !k->request) {
    fprintf(stderr, "  [eval] FAIL: invalid kernel handle\n");
    return false;
  }
  id mdl = (__bridge id)k->model;
  id req = (__bridge id)k->request;
  NSError *e = nil;
  BOOL ok = NO;
  if (k->clientEvalEnabled && k->client) {
    id client = (__bridge id)k->client;
    if ([client respondsToSelector:@selector
                (evaluateWithModel:options:request:qos:error:)]) {
      @try {
        ok = ((BOOL (*)(id, SEL, id, id, id, unsigned int,
                        NSError **))objc_msgSend)(
            client, @selector(evaluateWithModel:options:request:qos:error:),
            mdl, @{}, req, 21, &e);
        if (!ok)
          e = nil;
      } @catch (NSException *ex) {
        (void)ex;
        ok = NO;
      }
    }
  }
  if (!ok) {
    @try {
      ok = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
          mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req,
          &e);
    } @catch (NSException *ex) {
      (void)ex;
      ok = NO;
    }
    if (ok && k->clientEvalEnabled && !k->warnedClientFallback) {
      fprintf(stderr, "  [eval] _ANEClient path unavailable, using fallback\n");
      k->warnedClientFallback = 1;
    }
  }
  if (!ok && k->perfStats) {
    CFRelease(k->perfStats);
    k->perfStats = NULL;
    if (kern_rebuild_request(k)) {
      req = (__bridge id)k->request;
      @try {
        ok =
            ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{},
                req, &e);
      } @catch (NSException *ex) {
        (void)ex;
        ok = NO;
      }
      if (ok) {
        fprintf(stderr,
                "  [eval] recovered by disabling perfStats for this kernel\n");
      }
    }
  }
  if (!ok) {
    fprintf(stderr, "  [eval] FAIL: %s\n",
            e ? [[e description] UTF8String] : "no error");
    return false;
  }
  k->evalCount++;
  k->lastHwExecNS = 0;
  if (k->perfStats) {
    id perf = (__bridge id)k->perfStats;
    if ([perf respondsToSelector:@selector(hwExecutionTime)]) {
      uint64_t hwNS = ((uint64_t (*)(id, SEL))objc_msgSend)(perf, @selector
                                                            (hwExecutionTime));
      k->lastHwExecNS = hwNS;
      if (hwNS > 0) {
        k->totalHwExecNS += hwNS;
        k->perfSamples++;
      }
    }
  }
  return true;
}
