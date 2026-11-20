//! Unit tests for the layer class swap and the client surface field read.
//!
//! The swap is pinned on stand-ins for winemac's classes: a `CAMetalLayer`
//! subclass in the role of the hook class, which turns into a plain
//! `CAMetalLayer` exactly once, and another subclass, which keeps its class.
//! A real key-value observer pins that a layer is judged by the class it
//! reports, while the observer's subclass stays in place. The field read is
//! pinned on runtime-built classes carrying the field the way winemac's cocoa
//! view does, under either name a Wine gives it (with `CrossOver`'s preferred
//! when a class declares both), and on classes that lack it or carry it under
//! another type.

use core::ffi::c_void;

use objc2::{
    define_class, extern_methods,
    rc::Retained,
    runtime::{ClassBuilder, NSObject},
};
use objc2_foundation::{
    NSKeyValueObservingOptions, NSObjectNSKeyValueObserverRegistration, NSString,
};

use super::*;

define_class!(
    /// Stands in for winemac's `WineMetalLayer`.
    #[unsafe(super(CAMetalLayer))]
    #[name = "Mtld3dTestHookLayer"]
    struct HookLayer;
);

define_class!(
    /// A layer subclass that is not the hook class.
    #[unsafe(super(CAMetalLayer))]
    #[name = "Mtld3dTestOtherLayer"]
    struct OtherLayer;
);

impl HookLayer {
    extern_methods!(
        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        fn new() -> Retained<Self>;
    );
}

impl OtherLayer {
    extern_methods!(
        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        fn new() -> Retained<Self>;
    );
}

define_class!(
    /// A key-value observer that ignores what it is told.
    #[unsafe(super(NSObject))]
    #[name = "Mtld3dTestLayerObserver"]
    struct LayerObserver;

    impl LayerObserver {
        #[unsafe(method(observeValueForKeyPath:ofObject:change:context:))]
        fn observe_value(
            &self,
            _key_path: Option<&NSString>,
            _object: Option<&AnyObject>,
            _change: Option<&AnyObject>,
            _context: *mut c_void,
        ) {
        }
    }
);

impl LayerObserver {
    extern_methods!(
        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        fn new() -> Retained<Self>;
    );
}

/// A key-value observer on a layer's `contentsScale`, which gives the layer a subclass of its own.
struct Observation {
    observer: Retained<LayerObserver>,
    key: Retained<NSString>,
}

impl Observation {
    /// Register an observer on `layer`.
    fn new(layer: &CAMetalLayer) -> Self {
        let observer = LayerObserver::new();
        let key = NSString::from_str("contentsScale");
        // SAFETY: `observer` implements `observeValueForKeyPath:ofObject:change:context:`
        // and outlives the registration, which `remove` ends before the
        // layer drops; the context is null and never read.
        unsafe {
            layer.addObserver_forKeyPath_options_context(
                &observer,
                &key,
                NSKeyValueObservingOptions::empty(),
                core::ptr::null_mut(),
            );
        }
        Self { observer, key }
    }

    /// End the registration on `layer`, the layer [`Self::new`] observed.
    fn remove(self, layer: &CAMetalLayer) {
        // SAFETY: `new` registered this observer on `layer` for this key path.
        unsafe { layer.removeObserver_forKeyPath(&self.observer, &self.key) };
    }
}

/// The isa class of `layer`, which a key-value observer's subclass replaces.
fn class_of(layer: &CAMetalLayer) -> &'static AnyClass {
    let object: &AnyObject = layer.as_ref();
    object.class()
}

#[test]
fn the_hook_class_becomes_a_plain_layer_once() {
    let layer = HookLayer::new();
    let layer: &CAMetalLayer = &layer;
    assert_eq!(
        bypass_hook_class(layer, HookLayer::class()),
        LayerClass::Bypassed
    );
    assert_eq!(class_of(layer), CAMetalLayer::class());
    assert_eq!(
        bypass_hook_class(layer, HookLayer::class()),
        LayerClass::Plain,
        "a layer already bypassed is left alone"
    );
}

#[test]
fn a_plain_layer_stays_plain() {
    let layer = CAMetalLayer::new();
    assert_eq!(
        bypass_hook_class(&layer, HookLayer::class()),
        LayerClass::Plain
    );
    assert_eq!(class_of(&layer), CAMetalLayer::class());
}

#[test]
fn another_subclass_keeps_its_class() {
    let layer = OtherLayer::new();
    let layer: &CAMetalLayer = &layer;
    assert_eq!(
        bypass_hook_class(layer, HookLayer::class()),
        LayerClass::Foreign
    );
    assert_eq!(class_of(layer), OtherLayer::class());
}

#[test]
fn a_plain_layer_under_a_key_value_observer_is_plain() {
    let layer = CAMetalLayer::new();
    let observation = Observation::new(&layer);
    let observed = class_of(&layer);
    assert_ne!(
        observed,
        CAMetalLayer::class(),
        "the observer subclassed the layer"
    );
    assert_eq!(
        bypass_hook_class(&layer, HookLayer::class()),
        LayerClass::Plain
    );
    assert_eq!(
        class_of(&layer),
        observed,
        "the observer keeps its subclass"
    );
    observation.remove(&layer);
}

#[test]
fn the_hook_class_under_a_key_value_observer_keeps_its_class() {
    let layer = HookLayer::new();
    let layer: &CAMetalLayer = &layer;
    let observation = Observation::new(layer);
    let observed = class_of(layer);
    assert_ne!(
        observed,
        HookLayer::class(),
        "the observer subclassed the layer"
    );
    assert_eq!(
        bypass_hook_class(layer, HookLayer::class()),
        LayerClass::Foreign
    );
    assert_eq!(class_of(layer), observed, "the observer keeps its subclass");
    observation.remove(layer);
}

/// A runtime class named `name` over `superclass`, with the client surface field typed `T`.
///
/// The field carries `CrossOver`'s name, the first of [`CLIENT_SURFACE_IVARS`].
fn class_with_field<T: Encode>(name: &CStr, superclass: &AnyClass) -> &'static AnyClass {
    class_with_named_field::<T>(name, superclass, CLIENT_SURFACE_IVARS[0])
}

/// A runtime class named `name` over `superclass`, with a `T` field named `field`.
fn class_with_named_field<T: Encode>(
    name: &CStr,
    superclass: &AnyClass,
    field: &CStr,
) -> &'static AnyClass {
    let mut builder = ClassBuilder::new(name, superclass).expect("the test class name is free");
    builder.add_ivar::<T>(field);
    builder.register()
}

/// A runtime class named `name` over `superclass` with no fields of its own.
fn class_without_field(name: &CStr, superclass: &AnyClass) -> &'static AnyClass {
    ClassBuilder::new(name, superclass)
        .expect("the test class name is free")
        .register()
}

/// A zeroed instance of `class`.
fn instance(class: &AnyClass) -> Retained<AnyObject> {
    // SAFETY: `class` is a registered class; the runtime zeroes the instance,
    // and `Retained` takes the one reference creation returns.
    let raw = unsafe { objc2::ffi::class_createInstance(class, 0) };
    // SAFETY: as above.
    unsafe { Retained::from_raw(raw) }.expect("the runtime created the instance")
}

/// Write the client surface field of `object`, which is laid out as `host`.
fn set_field(object: &AnyObject, host: &AnyClass, surface: usize) {
    set_named_field(object, host, CLIENT_SURFACE_IVARS[0], surface);
}

/// Write the pointer field `field` of `object`, which is laid out as `host`.
fn set_named_field(object: &AnyObject, host: &AnyClass, field: &CStr, surface: usize) {
    let ivar = host
        .instance_variable(field)
        .expect("the host declares the field");
    // SAFETY: `object` is an instance of `host` or a subclass, which declares
    // the field.
    let field = unsafe { ivar.load_ptr::<*mut c_void>(object) };
    // SAFETY: the field is a pointer, `field` points into the live test
    // object, and nothing else touches it.
    unsafe { field.write(surface as *mut c_void) };
}

#[test]
fn the_field_names_the_surface_on_the_host_and_its_subclasses() {
    let host = class_with_field::<*mut c_void>(c"Mtld3dTestContentView", NSObject::class());
    let subclass = class_without_field(c"Mtld3dTestContentSubview", host);

    let view = instance(host);
    set_field(&view, host, 0x1234);
    assert_eq!(surface_field(&view, host), 0x1234);

    let derived = instance(subclass);
    set_field(&derived, host, 0x5678);
    assert_eq!(surface_field(&derived, host), 0x5678);

    let unset = instance(host);
    assert_eq!(surface_field(&unset, host), 0, "a view showing no surface");
}

#[test]
fn the_second_name_of_the_field_names_the_surface_too() {
    let host = class_with_named_field::<*mut c_void>(
        c"Mtld3dTestSecondNameContentView",
        NSObject::class(),
        CLIENT_SURFACE_IVARS[1],
    );
    let view = instance(host);
    set_named_field(&view, host, CLIENT_SURFACE_IVARS[1], 0x9abc);
    assert_eq!(surface_field(&view, host), 0x9abc);
}

#[test]
fn the_crossover_name_wins_when_a_class_declares_both() {
    let mut builder = ClassBuilder::new(c"Mtld3dTestBothContentView", NSObject::class())
        .expect("the test class name is free");
    builder.add_ivar::<*mut c_void>(CLIENT_SURFACE_IVARS[1]);
    builder.add_ivar::<*mut c_void>(CLIENT_SURFACE_IVARS[0]);
    let host = builder.register();
    let view = instance(host);
    set_named_field(&view, host, CLIENT_SURFACE_IVARS[1], 0x1111);
    set_named_field(&view, host, CLIENT_SURFACE_IVARS[0], 0x2222);
    assert_eq!(surface_field(&view, host), 0x2222);
}

#[test]
fn the_hook_is_replaced_only_when_a_surface_can_be_read_and_presented() {
    let first = class_with_field::<*mut c_void>(c"Mtld3dTestHookFirstName", NSObject::class());
    let second = class_with_named_field::<*mut c_void>(
        c"Mtld3dTestHookSecondName",
        NSObject::class(),
        CLIENT_SURFACE_IVARS[1],
    );
    let bare = class_without_field(c"Mtld3dTestHookBare", NSObject::class());
    let mistyped = class_with_field::<u32>(c"Mtld3dTestHookMistyped", NSObject::class());

    assert!(can_replace_hook(true, Some(first)));
    assert!(can_replace_hook(true, Some(second)));
    assert!(
        !can_replace_hook(false, Some(first)),
        "without win32u's exports nothing presents the surface"
    );
    assert!(
        !can_replace_hook(true, Some(bare)),
        "a class without the field names no surface"
    );
    assert!(
        !can_replace_hook(true, Some(mistyped)),
        "a field that is no pointer is not read"
    );
    assert!(
        !can_replace_hook(true, None),
        "a Wine without the cocoa view class"
    );
}

#[test]
fn an_object_outside_the_host_class_names_no_surface() {
    let host = class_with_field::<*mut c_void>(c"Mtld3dTestOtherContentView", NSObject::class());
    let unrelated = instance(NSObject::class());
    assert_eq!(surface_field(&unrelated, host), 0);
}

#[test]
fn a_host_without_a_pointer_field_names_no_surface() {
    let bare = class_without_field(c"Mtld3dTestBareContentView", NSObject::class());
    assert_eq!(surface_field(&instance(bare), bare), 0);

    let mistyped = class_with_field::<u32>(c"Mtld3dTestMistypedContentView", NSObject::class());
    assert_eq!(surface_field(&instance(mistyped), mistyped), 0);
}
