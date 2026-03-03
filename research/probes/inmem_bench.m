#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <objc/message.h>
#import <objc/runtime.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) {
  return (double)t * g_tb.numer / g_tb.denom / 1e6;
}
static const unsigned int kANEQoS = 21;
static const int kWarmupIters = 5;
static const int kBenchIters = 50;

static IOSurfaceRef makeSurface(NSUInteger bytes) {
  return IOSurfaceCreate((__bridge CFDictionaryRef) @{
    (id)kIOSurfaceWidth : @(bytes),
    (id)kIOSurfaceHeight : @1,
    (id)kIOSurfaceBytesPerElement : @1,
    (id)kIOSurfaceBytesPerRow : @(bytes),
    (id)kIOSurfaceAllocSize : @(bytes),
    (id)kIOSurfacePixelFormat : @0
  });
}

static id makeRequest(Class requestClass, Class ioClass, IOSurfaceRef ioIn,
                      IOSurfaceRef ioOut) {
  id wIn = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
      ioClass, @selector(objectWithIOSurface:), ioIn);
  id wOut = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
      ioClass, @selector(objectWithIOSurface:), ioOut);
  return ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
      requestClass,
      @selector(requestWithInputs:
                     inputIndices:outputs:outputIndices:weightsBuffer:perfStats
                                 :procedureIndex:),
      @[ wIn ], @[ @0 ], @[ wOut ], @[ @0 ], nil, nil, @0);
}

static void evaluateNTimes(id model, id request, int iters, NSError **error) {
  for (int i = 0; i < iters; i++) {
    ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        model, @selector(evaluateWithQoS:options:request:error:), kANEQoS, @{},
        request, error);
  }
}

double benchInMem(int ch, int sp) {
  @autoreleasepool {
    NSError *e = nil;
    NSString *path = [NSString
        stringWithFormat:@"/tmp/ane_sram_%dch_%dsp.mlpackage", ch, sp];
    NSURL *compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:path]
                                           error:&e];
    if (e)
      return -1;

    NSData *milData = [[NSString
        stringWithContentsOfFile:
            [[compiled path] stringByAppendingPathComponent:@"model.mil"]
                        encoding:NSUTF8StringEncoding
                           error:nil] dataUsingEncoding:NSUTF8StringEncoding];
    NSData *weightBlob = [NSData
        dataWithContentsOfFile:[[compiled path] stringByAppendingPathComponent:
                                                    @"weights/weight.bin"]];

    Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class IMM = NSClassFromString(@"_ANEInMemoryModel");
    Class AR = NSClassFromString(@"_ANERequest");
    Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");

    NSDictionary *wdict = @{
      @"@model_path/weights/weight.bin" :
          @{@"offset" : @64, @"data" : weightBlob}
    };
    id desc = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
        Desc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict,
        nil);
    if (!desc)
      return -2;
    id model = ((id (*)(Class, SEL, id))objc_msgSend)(
        IMM, @selector(inMemoryModelWithDescriptor:), desc);
    if (!model)
      return -3;

    id hexId =
        ((id (*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString *tmpDir =
        [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES
                         attributes:nil
                              error:nil];
    [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"]
              atomically:YES];
    [weightBlob
        writeToFile:[tmpDir
                        stringByAppendingPathComponent:@"weights/weight.bin"]
         atomically:YES];

    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), kANEQoS, @{}, &e);
    if (!ok) {
      [fm removeItemAtPath:tmpDir error:nil];
      return -4;
    }

    ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), kANEQoS, @{}, &e);
    if (!ok) {
      [fm removeItemAtPath:tmpDir error:nil];
      return -5;
    }

    NSUInteger bytes = ch * sp * 4;
    IOSurfaceRef ioIn = makeSurface(bytes);
    IOSurfaceRef ioOut = makeSurface(bytes);
    id req = makeRequest(AR, AIO, ioIn, ioOut);

    evaluateNTimes(model, req, kWarmupIters, &e);

    uint64_t t0 = mach_absolute_time();
    evaluateNTimes(model, req, kBenchIters, &e);
    double ms = ticksToMs(mach_absolute_time() - t0) / kBenchIters;

    ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
        model, @selector(unloadWithQoS:error:), kANEQoS, &e);
    CFRelease(ioIn);
    CFRelease(ioOut);
    [fm removeItemAtPath:tmpDir error:nil];
    return ms;
  }
}

int main() {
  mach_timebase_info(&g_tb);
  dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
         "AppleNeuralEngine",
         RTLD_NOW);

  printf("=== In-Memory ANE Benchmark ===\n\n");
  printf("%-12s %8s %10s %8s\n", "Config", "W (MB)", "ms/eval", "TFLOPS");
  printf("---------------------------------------------\n");

  int chs[] = {256, 512, 1024, 2048, 3072, 4096};
  int sps[] = {64, 64, 64, 64, 64, 64};
  for (int i = 0; i < 6; i++) {
    int ch = chs[i], sp = sps[i];
    double w_mb = (double)ch * ch * 2 / 1024 / 1024;
    double gf = 2.0 * ch * ch * sp / 1e9;
    double ms = benchInMem(ch, sp);
    double tflops = (ms > 0) ? gf / ms : 0;
    if (ms > 0)
      printf("%4dch x%2dsp %7.1f  %8.3f ms %7.2f\n", ch, sp, w_mb, ms, tflops);
    else
      printf("%4dch x%2dsp %7.1f  FAIL(%.0f)\n", ch, sp, w_mb, ms);
  }
  return 0;
}
