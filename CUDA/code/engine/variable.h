#pragma once

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>


#include "errors.h"
class EmptyClass {};

class Variable {
 public:
  template <typename T>
  const T& Get() const {
    ENFORCE_NOT_NULL(
        holder_, "Variable is not initialized.");
    ENFORCE_EQ(
        holder_->Type(),
        std::type_index(typeid(T)),
        InvalidArgument("The Variable type must be %s, but the type it holds is %s.",
            std::type_index(typeid(T)).name(),
            holder_->Type().name()));
    return *static_cast<const T*>(holder_->Ptr());
  }

  bool IsInitialized() const { return holder_ != nullptr; }

  template <typename T>
  T* GetMutable() {
    if (!holder_) {
      holder_.reset(new PlaceholderImpl<T>());
    } else {
      ENFORCE_EQ(
          holder_->Type(),
          std::type_index(typeid(T)),
          InvalidArgument("The Variable type must be %s, but the type it holds is %s.",
              std::type_index(typeid(T)).name(),
              holder_->Type().name()));
    }
    return static_cast<T*>(holder_->Ptr());
  }

  template <typename T>
  T* Reset(T* t) {
    if (!holder_) {
      holder_.reset(new PlaceholderImpl<T>());
    } else {
      ENFORCE_EQ(
          holder_->Type(),
          std::type_index(typeid(T)),
          InvalidArgument(
              "The reset type (%s) must same as the type it holds (%s).",
              std::type_index(typeid(T)).name(),
              holder_->Type().name()));
      holder_->Reset(static_cast<void*>(t));
    }
    return static_cast<T*>(holder_->Ptr());
  }

  template <typename T>
  bool IsType() const {
    return holder_ && holder_->Type() == std::type_index(typeid(T));
  }

  void Clear() { holder_.reset(); }

  const std::type_index& Type() const {
    ENFORCE_NOT_NULL(
        holder_, "Variable is not initialized.");
    return holder_->Type();
  }

 private:
  struct Placeholder {
    Placeholder() : type_(typeid(EmptyClass)) { }

    virtual ~Placeholder() {}

    inline const std::type_index& Type() const { return type_; }
    inline const void* Ptr() const { return ptr_; }
    inline void* Ptr() { return ptr_; }
    inline void Reset(void* p) { ptr_ = p; }

   protected:
    inline void Init(void* p, const std::type_index& type) {
      ptr_ = p;
      type_ = type;
    }

    void* ptr_;
    std::type_index type_;
  };

  // Placeholder hides type T, so it doesn't appear as a template
  // parameter of Variable.
  template <typename T>
  struct PlaceholderImpl : public Placeholder {
    PlaceholderImpl() : Placeholder() { this->Init((void*)&obj_, std::type_index(typeid(obj_))); }

   private:
    T obj_;
  };

  // pointers to a PlaceholderImpl object indeed.
  std::shared_ptr<Placeholder> holder_;
};