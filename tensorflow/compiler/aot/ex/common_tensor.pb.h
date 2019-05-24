// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: common_tensor.proto

#ifndef PROTOBUF_INCLUDED_common_5ftensor_2eproto
#define PROTOBUF_INCLUDED_common_5ftensor_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3007000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3007001 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_common_5ftensor_2eproto

// Internal implementation detail -- do not use these members.
struct TableStruct_common_5ftensor_2eproto {
  static const ::google::protobuf::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::google::protobuf::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors_common_5ftensor_2eproto();
class CommonTensor;
class CommonTensorDefaultTypeInternal;
extern CommonTensorDefaultTypeInternal _CommonTensor_default_instance_;
class CommonTensors;
class CommonTensorsDefaultTypeInternal;
extern CommonTensorsDefaultTypeInternal _CommonTensors_default_instance_;
namespace google {
namespace protobuf {
template<> ::CommonTensor* Arena::CreateMaybeMessage<::CommonTensor>(Arena*);
template<> ::CommonTensors* Arena::CreateMaybeMessage<::CommonTensors>(Arena*);
}  // namespace protobuf
}  // namespace google

// ===================================================================

class CommonTensor :
    public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:CommonTensor) */ {
 public:
  CommonTensor();
  virtual ~CommonTensor();

  CommonTensor(const CommonTensor& from);

  inline CommonTensor& operator=(const CommonTensor& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  CommonTensor(CommonTensor&& from) noexcept
    : CommonTensor() {
    *this = ::std::move(from);
  }

  inline CommonTensor& operator=(CommonTensor&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor() {
    return default_instance().GetDescriptor();
  }
  static const CommonTensor& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const CommonTensor* internal_default_instance() {
    return reinterpret_cast<const CommonTensor*>(
               &_CommonTensor_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(CommonTensor* other);
  friend void swap(CommonTensor& a, CommonTensor& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline CommonTensor* New() const final {
    return CreateMaybeMessage<CommonTensor>(nullptr);
  }

  CommonTensor* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<CommonTensor>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const CommonTensor& from);
  void MergeFrom(const CommonTensor& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  static const char* _InternalParse(const char* begin, const char* end, void* object, ::google::protobuf::internal::ParseContext* ctx);
  ::google::protobuf::internal::ParseFunc _ParseFunc() const final { return _InternalParse; }
  #else
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(CommonTensor* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated int32 shape = 2;
  int shape_size() const;
  void clear_shape();
  static const int kShapeFieldNumber = 2;
  ::google::protobuf::int32 shape(int index) const;
  void set_shape(int index, ::google::protobuf::int32 value);
  void add_shape(::google::protobuf::int32 value);
  const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
      shape() const;
  ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
      mutable_shape();

  // repeated float data = 3;
  int data_size() const;
  void clear_data();
  static const int kDataFieldNumber = 3;
  float data(int index) const;
  void set_data(int index, float value);
  void add_data(float value);
  const ::google::protobuf::RepeatedField< float >&
      data() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_data();

  // string name = 1;
  void clear_name();
  static const int kNameFieldNumber = 1;
  const ::std::string& name() const;
  void set_name(const ::std::string& value);
  #if LANG_CXX11
  void set_name(::std::string&& value);
  #endif
  void set_name(const char* value);
  void set_name(const char* value, size_t size);
  ::std::string* mutable_name();
  ::std::string* release_name();
  void set_allocated_name(::std::string* name);

  // @@protoc_insertion_point(class_scope:CommonTensor)
 private:
  class HasBitSetters;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedField< ::google::protobuf::int32 > shape_;
  mutable std::atomic<int> _shape_cached_byte_size_;
  ::google::protobuf::RepeatedField< float > data_;
  mutable std::atomic<int> _data_cached_byte_size_;
  ::google::protobuf::internal::ArenaStringPtr name_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_common_5ftensor_2eproto;
};
// -------------------------------------------------------------------

class CommonTensors :
    public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:CommonTensors) */ {
 public:
  CommonTensors();
  virtual ~CommonTensors();

  CommonTensors(const CommonTensors& from);

  inline CommonTensors& operator=(const CommonTensors& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  CommonTensors(CommonTensors&& from) noexcept
    : CommonTensors() {
    *this = ::std::move(from);
  }

  inline CommonTensors& operator=(CommonTensors&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor() {
    return default_instance().GetDescriptor();
  }
  static const CommonTensors& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const CommonTensors* internal_default_instance() {
    return reinterpret_cast<const CommonTensors*>(
               &_CommonTensors_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  void Swap(CommonTensors* other);
  friend void swap(CommonTensors& a, CommonTensors& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline CommonTensors* New() const final {
    return CreateMaybeMessage<CommonTensors>(nullptr);
  }

  CommonTensors* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<CommonTensors>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const CommonTensors& from);
  void MergeFrom(const CommonTensors& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  static const char* _InternalParse(const char* begin, const char* end, void* object, ::google::protobuf::internal::ParseContext* ctx);
  ::google::protobuf::internal::ParseFunc _ParseFunc() const final { return _InternalParse; }
  #else
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(CommonTensors* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .CommonTensor data = 1;
  int data_size() const;
  void clear_data();
  static const int kDataFieldNumber = 1;
  ::CommonTensor* mutable_data(int index);
  ::google::protobuf::RepeatedPtrField< ::CommonTensor >*
      mutable_data();
  const ::CommonTensor& data(int index) const;
  ::CommonTensor* add_data();
  const ::google::protobuf::RepeatedPtrField< ::CommonTensor >&
      data() const;

  // @@protoc_insertion_point(class_scope:CommonTensors)
 private:
  class HasBitSetters;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedPtrField< ::CommonTensor > data_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_common_5ftensor_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// CommonTensor

// string name = 1;
inline void CommonTensor::clear_name() {
  name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& CommonTensor::name() const {
  // @@protoc_insertion_point(field_get:CommonTensor.name)
  return name_.GetNoArena();
}
inline void CommonTensor::set_name(const ::std::string& value) {
  
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:CommonTensor.name)
}
#if LANG_CXX11
inline void CommonTensor::set_name(::std::string&& value) {
  
  name_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:CommonTensor.name)
}
#endif
inline void CommonTensor::set_name(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:CommonTensor.name)
}
inline void CommonTensor::set_name(const char* value, size_t size) {
  
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:CommonTensor.name)
}
inline ::std::string* CommonTensor::mutable_name() {
  
  // @@protoc_insertion_point(field_mutable:CommonTensor.name)
  return name_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* CommonTensor::release_name() {
  // @@protoc_insertion_point(field_release:CommonTensor.name)
  
  return name_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void CommonTensor::set_allocated_name(::std::string* name) {
  if (name != nullptr) {
    
  } else {
    
  }
  name_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), name);
  // @@protoc_insertion_point(field_set_allocated:CommonTensor.name)
}

// repeated int32 shape = 2;
inline int CommonTensor::shape_size() const {
  return shape_.size();
}
inline void CommonTensor::clear_shape() {
  shape_.Clear();
}
inline ::google::protobuf::int32 CommonTensor::shape(int index) const {
  // @@protoc_insertion_point(field_get:CommonTensor.shape)
  return shape_.Get(index);
}
inline void CommonTensor::set_shape(int index, ::google::protobuf::int32 value) {
  shape_.Set(index, value);
  // @@protoc_insertion_point(field_set:CommonTensor.shape)
}
inline void CommonTensor::add_shape(::google::protobuf::int32 value) {
  shape_.Add(value);
  // @@protoc_insertion_point(field_add:CommonTensor.shape)
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
CommonTensor::shape() const {
  // @@protoc_insertion_point(field_list:CommonTensor.shape)
  return shape_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
CommonTensor::mutable_shape() {
  // @@protoc_insertion_point(field_mutable_list:CommonTensor.shape)
  return &shape_;
}

// repeated float data = 3;
inline int CommonTensor::data_size() const {
  return data_.size();
}
inline void CommonTensor::clear_data() {
  data_.Clear();
}
inline float CommonTensor::data(int index) const {
  // @@protoc_insertion_point(field_get:CommonTensor.data)
  return data_.Get(index);
}
inline void CommonTensor::set_data(int index, float value) {
  data_.Set(index, value);
  // @@protoc_insertion_point(field_set:CommonTensor.data)
}
inline void CommonTensor::add_data(float value) {
  data_.Add(value);
  // @@protoc_insertion_point(field_add:CommonTensor.data)
}
inline const ::google::protobuf::RepeatedField< float >&
CommonTensor::data() const {
  // @@protoc_insertion_point(field_list:CommonTensor.data)
  return data_;
}
inline ::google::protobuf::RepeatedField< float >*
CommonTensor::mutable_data() {
  // @@protoc_insertion_point(field_mutable_list:CommonTensor.data)
  return &data_;
}

// -------------------------------------------------------------------

// CommonTensors

// repeated .CommonTensor data = 1;
inline int CommonTensors::data_size() const {
  return data_.size();
}
inline void CommonTensors::clear_data() {
  data_.Clear();
}
inline ::CommonTensor* CommonTensors::mutable_data(int index) {
  // @@protoc_insertion_point(field_mutable:CommonTensors.data)
  return data_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::CommonTensor >*
CommonTensors::mutable_data() {
  // @@protoc_insertion_point(field_mutable_list:CommonTensors.data)
  return &data_;
}
inline const ::CommonTensor& CommonTensors::data(int index) const {
  // @@protoc_insertion_point(field_get:CommonTensors.data)
  return data_.Get(index);
}
inline ::CommonTensor* CommonTensors::add_data() {
  // @@protoc_insertion_point(field_add:CommonTensors.data)
  return data_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::CommonTensor >&
CommonTensors::data() const {
  // @@protoc_insertion_point(field_list:CommonTensors.data)
  return data_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)


// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // PROTOBUF_INCLUDED_common_5ftensor_2eproto
