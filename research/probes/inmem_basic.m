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
static const int kWarmupIters = 10;
static const int kBenchIters = 100;

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

static BOOL evaluateOnce(id model, id request, NSError **error) {
  return ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
      model, @selector(evaluateWithQoS:options:request:error:), kANEQoS, @{},
      request, error);
}

static void evaluateNTimes(id model, id request, int iters, NSError **error) {
  for (int i = 0; i < iters; i++) {
    (void)evaluateOnce(model, request, error);
  }
}

int main() {
  @autoreleasepool {
    mach_timebase_info(&g_tb);
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
           "AppleNeuralEngine",
           RTLD_NOW);

    NSError *e = nil;
    int ch = 256, sp = 64;

    // Get MIL and weights from a compiled model
    NSURL *compiled = [MLModel
        compileModelAtURL:
            [NSURL fileURLWithPath:@"/tmp/ane_sram_256ch_64sp.mlpackage"]
                    error:&e];
    if (e) {
      printf("Compile failed\n");
      return 1;
    }

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
    id model = ((id (*)(Class, SEL, id))objc_msgSend)(
        IMM, @selector(inMemoryModelWithDescriptor:), desc);

    // Get the hex identifier to pre-populate the temp dir
    id hexId =
        ((id (*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString *tmpDir =
        [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
    NSFileManager *fm = [NSFileManager defaultManager];

    // Pre-create dir with MIL and weights
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
    printf("Pre-created: %s\n", [tmpDir UTF8String]);

    // Compile
    printf("Compiling...\n");
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), kANEQoS, @{}, &e);
    printf("compile: %s\n", ok ? "YES" : "NO");
    if (e) {
      printf("  err: %s\n", [[e description] UTF8String]);
      e = nil;
    }
    if (!ok)
      return 1;

    // Load
    printf("Loading...\n");
    ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), kANEQoS, @{}, &e);
    printf("load: %s\n", ok ? "YES" : "NO");
    if (e) {
      printf("  err: %s\n", [[e description] UTF8String]);
      e = nil;
    }
    if (!ok)
      return 1;

    printf("state: %lu\n",
           ((NSUInteger (*)(id, SEL))objc_msgSend)(model, @selector(state)));

    // Create IO surfaces
    NSUInteger bytes = ch * sp * 4;
    IOSurfaceRef ioIn = makeSurface(bytes);
    IOSurfaceRef ioOut = makeSurface(bytes);
    id req = makeRequest(AR, AIO, ioIn, ioOut);

    // Evaluate
    printf("Evaluating...\n");
    ok = evaluateOnce(model, req, &e);
    printf("evaluate: %s\n", ok ? "YES" : "NO");
    if (e) {
      printf("  err: %s\n", [[e description] UTF8String]);
      e = nil;
    }

    if (ok) {
      // Warmup
      evaluateNTimes(model, req, kWarmupIters, &e);

      // Benchmark
      uint64_t t0 = mach_absolute_time();
      evaluateNTimes(model, req, kBenchIters, &e);
      double ms = ticksToMs(mach_absolute_time() - t0) / kBenchIters;
      double gf = 2.0 * ch * ch * sp / 1e9;
      double tflops = gf / ms;

      printf("\n========================================\n");
      printf("IN-MEMORY ANE EXECUTION SUCCESSFUL!\n");
      printf("  %.3f ms/eval, %.2f TFLOPS\n", ms, tflops);
      printf("========================================\n");
    }

    // Cleanup
    ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
        model, @selector(unloadWithQoS:error:), kANEQoS, &e);
    CFRelease(ioIn);
    CFRelease(ioOut);
    [fm removeItemAtPath:tmpDir error:nil];
  }
  return 0;
}
