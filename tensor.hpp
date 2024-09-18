#pragma once
#include <tuple>
#include <ostream>
#include <istream>
#include <fstream>
#include <random>
#include <functional>
#include <array>
#include <stacktrace>
#include <variant>
#include <cassert>
#include <filesystem>
#include <vector>

#include "timer.hpp"
#include "fmt/format.h"
#include "fmt/color.h"

namespace fs = std::filesystem;

std::mt19937 g_randomEngine(213);

int readNumber_(std::istream& is) {
	int r;
	is >> r;
	is.ignore(1);
	return r;
}

// Reads the first integer from the stream, consumes only one following char
int readNumber(std::istream& is) {
	//std::stringstream is("2 11 22 234234");

	//auto pos0 = is.tellg();

	char c;
	int r = 0;
	bool done = false;
	while (is.get(c) && '0' <= c && c <= '9') {
		r = r*10 + (c - '0');
		done = true;
	}
	assert(done);

	//auto pos1 = is.tellg();
	//is.seekg(pos0);

	//auto r0 = readNumber_(is);

	//auto pos2 = is.tellg();

	//assert(r == r0);
	//assert(pos1 == pos2);

	return r;
}


template<std::size_t N, typename... T>
using static_switch = typename std::tuple_element<N, std::tuple<T...> >::type;


// A tuple of integers to represent the shape of a tensor. 
struct tensorShape {
private:
	static const int max_dim = 8;
	std::array<int, max_dim> dims;
	int ndims;
public:
	constexpr tensorShape() : ndims{0}{}
	constexpr tensorShape(std::initializer_list<int>&& il) {
		for (ndims = 0; auto& a : il)
			dims[ndims++] = a;
	}
	constexpr explicit tensorShape(int numDimensions, int value=0) : ndims{ numDimensions } {
		dims.fill(value);
	}
	constexpr inline int size() const {
		return ndims;
	}
	constexpr inline void resize(int s) {
		ndims = s;
	}
	constexpr inline auto& operator[](this auto&& self, int i) {
		if (i < 0)  i += self.ndims;
		return self.dims[i];
	}
	constexpr inline auto& back(this auto&& self) {
		return self.dims[self.ndims-1];
	}
	constexpr int pop_back() {
		return dims[--ndims];
	}
	constexpr tensorShape pop_back(int n) {
		auto last = tensorShape(n);
		std::copy(dims.data() + ndims - n, dims.data() + ndims, last.dims.data());
		ndims -= n;
		return last;
	}
	constexpr void push_back(int d) {
#if defined _DEBUG
		if (ndims+1 > tensorShape::max_dim) throw std::runtime_error("Tensor shape is full.");
#endif
		dims[ndims++] = d;
	}
	constexpr void push_back(tensorShape const& ts) {
#if defined _DEBUG
		if (ndims+ts.size() > tensorShape::max_dim) throw std::runtime_error("Tensor shape is full.");
#endif
		std::copy(ts.dims.data(), ts.dims.data() + ts.ndims, dims.data() + ndims);
		ndims += ts.ndims;
	}
	constexpr auto& front(this auto&& self) {
		return self.dims[0];
	}
	constexpr int pop_front() {
		std::rotate(dims.begin(), dims.begin() + 1, dims.begin()+ndims);
		return dims[--ndims];
	}
	constexpr void contract(int d0, int d1) {
		int p = dims[d0]; for (int d = d0 + 1; d <= d1; ++d) p *= dims[d];
		dims[d0] = p;
		for (int d = d0 + 1, s = d1 - d0; d <= d1; ++d, --ndims) dims[d] = dims[d + s];
	}
	constexpr tensorShape tail() const {
		tensorShape other = *this;
		other.pop_front();
		return other;
	}
	constexpr int volume() const {
		if (ndims == 0) return 0;
		int i = 1;
		for (int j = 0; j < ndims; ++j) i *= dims[j];
		return i;
	}
	constexpr std::string toString() const {
		if (ndims == 0) return "<empty>";
		std::string ss = std::to_string(dims[0]);
		for (int j = 1; j < ndims; ++j) ss += "x" + std::to_string(dims[j]);
		return ss;
	}
	constexpr friend bool operator==(tensorShape const& l, tensorShape const& r) {
		return l.ndims == r.ndims && 0 == memcmp(&l, &r, (char*)&l.dims[l.ndims] - (char*)&l);
	}
	constexpr friend tensorShape operator*(tensorShape const& l, tensorShape const& r) {
		// Shape of tensor product.
		tensorShape ret = l;
		ret.push_back(r);
		return ret;
	}
	template<typename... T>
	constexpr friend tensorShape assembleShape(T const&... t) {
		tensorShape ts;
		(ts.push_back(t), ...);
		return ts;
	}
	/*constexpr void remove(int elem) {
		auto beg = std::begin(dims);
		ndims = std::distance(beg, std::remove(beg, beg + ndims, elem));
	}*/

	auto begin(this auto&& self) {
		return self.dims.begin();
	}
	auto end(this auto&& self) {
		return self.dims.begin() + self.ndims;
	}
	void setNDims(int n) {
		ndims = n;
	}
};


using index_t = tensorShape;
using uniform = std::uniform_real_distribution<float>;
using initializer = std::variant<uniform, float>;
using shapeOrScalar = std::variant<tensorShape, int>;

float getValue(initializer& v) {
	if (v.index() == 0) {
		return std::get<uniform>(v)(g_randomEngine);
	}
	else {
		return std::get<float>(v);
	}
}
tensorShape getShape(shapeOrScalar const& v, int dims) {
	if (v.index() == 0) {
		return std::get<tensorShape>(v);
	}
	else {
		return tensorShape(dims, std::get<int>(v));
	}
}


struct shapeError : public std::exception {
private:
	std::stacktrace stacktrace;

	static std::string getFilename(std::string path) {
		auto loc = path.find_last_of("/\\");
		if (loc == std::string::npos)
			return "";
		return path.substr(loc + 1);
	}
	static std::string getFuncname(std::string compilerFuncName) {
		auto loc0 = compilerFuncName.find('!');
		auto loc1 = compilerFuncName.find('+');
		return compilerFuncName.substr(loc0 + 1, loc1 - loc0 - 1);
	}

public:
	shapeError(char const* const msg, tensorShape const& expected, tensorShape const& actual)
		: std::exception(fmt::format("{} Expected {}, got {}.", msg, expected.toString(), actual.toString()).c_str()),
		stacktrace{ std::stacktrace::current() } {
	}
	shapeError(char const* const msg, int expectedNDims, tensorShape const& actual)
		: std::exception(fmt::format("{} Expected {} dimensions, got {}.", msg, expectedNDims, actual.toString()).c_str()),
		stacktrace{ std::stacktrace::current() } {
	}
	shapeError(char const* const msg, int expectedValue, int actualValue)
		: std::exception(fmt::format("{} Expected {}, got {}.", msg, expectedValue, actualValue).c_str()),
		stacktrace{ std::stacktrace::current() } {
	}
	std::string prettyStacktrace() const {
		AutoTimer at(g_timer, _FUNC_);
		auto col = fmt::color::slate_gray;
		std::string ret = fmt::format(fg(col), "Stacktrace:\n");
		for (int i = -1; auto & a : stacktrace) {
			if (++i == 0) continue;
			std::string funcName = getFuncname(a.description());
			ret += fmt::format(fg(col), "-> {:<25} {}\n",
				getFilename(a.source_file()) + ":" + std::to_string(a.source_line()),
				funcName);
			if (funcName == "main")
				break;
		}
		return ret;
	}
};


#if defined _DEBUG 
#define throw_if_value_mismatch(expectedValue, actualValue, msg) if (actualValue != expectedValue) throw shapeError(msg, expectedValue, actualValue)
#define throw_if_shape_mismatch(expectedShape, actualShape, msg) if (actualShape != expectedShape) throw shapeError(msg, expectedShape, actualShape)
#define throw_if_ndims_mismatch(expectedNDims, actualShape, msg) if (actualShape.size() != expectedNDims) throw shapeError(msg, expectedNDims, actualShape)
#define throw_if_ndims_smaller(expectedNDims, actualShape, msg) if (actualShape.size() < expectedNDims) throw shapeError(msg, expectedNDims, actualShape)
#define throw_if_ndims_greater(expectedNDims, actualShape, msg) if (actualShape.size() > expectedNDims) throw shapeError(msg, expectedNDims, actualShape)
#define throw_if_out_of_bounds(bounds, index, msg) \
		if(index.size() != bounds.size()) assert(false), throw std::invalid_argument("Dimensions do not match."); \
		for (int i = 0; i < index.size(); ++i) if(index[i] < 0 || index[i] >= bounds[i]) assert(false), throw std::out_of_range("Out of bounds " + std::to_string(i));

#else
#define throw_if_value_mismatch(expectedValue, actualValue, msg)
#define throw_if_shape_mismatch(expectedShape, actualShape, msg)
#define throw_if_ndims_mismatch(expectedShape, actualShape, msg)
#define throw_if_ndims_smaller(expectedNDims, actualShape, msg)
#define throw_if_ndims_greater(expectedNDims, actualShape, msg)
#define throw_if_out_of_bounds(bounds, index, msg)
#endif

inline int g_numAllocs = 0;

class TensorData {
	bool _owner = true;
	float* _data = nullptr;
	int _size = 0, _capacity = 0;

public:
	TensorData(){}
	TensorData(std::nullptr_t) { }
	// Full copy
	TensorData(auto&& begin, auto&& end) {
		_owner = true;
		resize(std::distance(begin, end));
		std::memcpy(_data, &(*begin), _size * sizeof(float));
	}
	// Weak copy
	TensorData(TensorData const& other) {
		*this = other;
	}
	TensorData(TensorData&& other) noexcept {
		*this = std::forward<TensorData>(other);
	}
	~TensorData() {
		if (_owner && _data) {
			//fmt::print(fg(fmt::color::orange_red), "delete\n");
			delete[] _data;
			_data = nullptr;
		}
	}
	TensorData& operator=(TensorData const& other) {
		_data = other._data;
		_size = other._size;
		_capacity = other._capacity;
		_owner = !other._data;
		return *this;
	}
	TensorData& operator=(TensorData&& other) noexcept {
		std::swap(_data, other._data);
		_size = other._size;
		_capacity = other._capacity;
		//_owner = other._owner;
		std::swap(_owner, other._owner);
		return *this;
	}

	auto* data(this auto&& self) {
		return self._data;
	}
	int   size() const { return _size; }
	bool  isnull() const { return !_data; }
	void  resize(int newSize) {
		if (!_owner)
			throw std::logic_error("Tensor data that act as a view cannot resize.");
		if (newSize > _capacity) {
			_capacity = newSize;
			//fmt::print(fg(fmt::color::chartreuse), "new\n");
			float* p = new float[_capacity];
			std::copy(_data, _data + _size, p);
			delete[] _data;
			_data = p;

			g_numAllocs++;
			//AutoTimer at(g_timer, _FUNC_);
		}
		_size = newSize;
	}

	void  fill(initializer const& init) {
		auto stateful = init;
		for (int i = 0; i < _size; ++i)
			_data[i] = getValue(stateful);
	}
	
	// Deep copy
	TensorData copy() const {
		TensorData td = *this;
		td._capacity = _size;

		td._data = new float[td._capacity];
		std::copy(_data, _data + _size, td._data);
		td._owner = true;

		g_numAllocs++;
		//AutoTimer at(g_timer, _FUNC_);

		return td;
	}

	TensorData subset(int offsetElems, int newSize) const {
		TensorData td = *this;
		td._owner = false;
		td._data += offsetElems;
		td._size = newSize;
		td._capacity -= offsetElems;
		return td;
	}
	inline auto& operator[](this auto&& self, int idx) {
#if defined _DEBUG
		if (idx < 0 || idx >= self._size)
			throw std::out_of_range("TensorData access out of range: " + std::to_string(idx) + ".");
#endif
		return self._data[idx];
	}
};


// A linear tensor iterator
class Tensor;
template<bool CONST = false>
struct TensorIterator {
	static_switch<CONST,Tensor*,const Tensor*> data;
	int i;
	auto operator<=>(const TensorIterator&) const = default;
	TensorIterator& operator++() {
		++i;
		return *this;
	}
	auto& operator*();
};

// A tensor is a view on data. It can own the data or borrow it.
class Tensor {
	friend class Tensor;
	friend struct fmt::formatter<Tensor>;

private:
	TensorData data;
	tensorShape shape;
	tensorShape strides;
	int offset;
	bool defaultStride = true;

	// Structured print.
	template<typename F>
	std::string toString(F&& f, std::string separator = ", ") const {
		int v = shape.volume();
		int d = shape.size();
		tensorShape fa;
		for (int x = 1, j = d - 1; j >= 0; --j)
			fa.push_back(x *= shape[j]);

		std::string ret, s;
		for (int p = 0, i = 0; i < v; ++i) {
			s.clear();
			for (int j = 0; j < d; ++j)
				if (i % fa[j] == 0)
					s += '{', ++p;
			ret += s + f((*this)[i]);
			s.clear();
			int nn = 0;
			for (int j = 0; j < d; ++j)
				if (i % fa[j] == fa[j] - 1)
					s += '}', --p, nn += !j;
			for (int o = 0; o < nn; ++o)
				s += (i == v - 1 ? "\n" : ",\n") + std::string(p, ' ');
			ret += (s.empty() ? separator : s);
		}
		return ret.substr(0, ret.size() - 1);
	}
	/*template<typename F>
	std::string toString(F&& f, std::string separator = ", ") const {
		int n = shape.volume();
		int d = shape.size();
		tensorShape fa;
		for (int x = 1, j = d - 1; j >= 0; --j)
			fa.push_back(x *= shape[j]);
		std::string ret, s;
		for (int p = 0, i = 0; i < n; ++i) {
			s.clear();
			for (int j = 0; j < d; ++j)
				if (i % fa[j] == 0)
					s += '{', ++p;
			ret += s + f(data[i]);
			s.clear();
			int nn = 0;
			for (int j = 0; j < d; ++j)
				if (i % fa[j] == fa[j] - 1)
					s += '}', --p, nn += !j;
			for (int o = 0; o < nn; ++o)
				s += (i == n - 1 ? "\n" : ",\n") + std::string(p, ' ');
			ret += (s.empty() ? separator : s);
		}
		return ret.substr(0, ret.size() - 1);
	}*/

	// Sets the stride to the natural value for the current shape
	void updateStridesFromShape() {
		int d = shape.size();
		strides.resize(d);
		for (int i = d - 1, l=1; i>=0; --i)
			strides[i] = l, l *= shape[i];
		offset = 0;
		checkShapeStrideConsistency();
		defaultStride = true;
	}
	void checkShapeStrideConsistency() const {
		int v = data.size();
		if (!v)
			return;
		int m = offset;
		for (int i = shape.size() - 1; i >= 0; --i)
			m += (shape[i]-1) * strides[i];
		if (m > v - 1)
			throw std::runtime_error(fmt::format("Shape {} with strides {} would access beyond data size {}.\n",
				shape.toString(), strides.toString(), v));
	}

	// Returns the index into the underlying data.
	/*inline int toLinear(tensorShape const& idx) const {
		int l = 0;
		for (int i = 0; i < idx.size(); ++i) l += strides[i] * idx[i];
		return l;
	}*/
	inline int toLinear(tensorShape const& idx) const {
		int l = offset, k = 1;
		for (int i = idx.size() - 1; i >= 0; --i)
			l += idx[i] * strides[i];
		return l;
		/*int l = 0, k = 1;
		for (int i = idx.size() - 1; i >= 0; --i)
			l = l * k + idx[i], k = strides[i];
		return l + offset;*/
	}	

public:
	Tensor()  {}
	// Creates a tensor with new data and sets value in each component.
	Tensor(tensorShape const& s, initializer value = 0.f) : shape{ s } {
		data.resize(s.volume()); data.fill(value);
		updateStridesFromShape();
	}
	// Creates a tensor for existing data.
	Tensor(tensorShape const& s, TensorData&& h) : shape{ s }, data{ std::forward<TensorData>(h) } { // TODO move?
		if(!h.isnull() && s.volume() != h.size())
			throw std::invalid_argument("The shape is not applicable to this data.");
		updateStridesFromShape();
	}

	// Creates one-dimensional tensor from float vector
	explicit Tensor(std::vector<float> const& v) : Tensor(tensorShape{(int)v.size()},TensorData(v.begin(),v.end())) {
	}
	 
	// Default shallow copy and move:
	Tensor(Tensor const& t) = default;
	Tensor(Tensor&& t) = default;
	Tensor& operator=(Tensor const& other) = default;
	Tensor& operator=(Tensor&& other) = default;

	// Deep copy
	Tensor copy() const {
		Tensor twin = *this;
		twin.data = data.copy();
		return twin;
	}

	enum class resizeMode {
		eDontAllowResize = 0,
		eResizeIfData,
		eForceResize
	};

	// At any moment the data should be zero or its size should fit the tensorShape. Resets the stride.
	void reshape(tensorShape const& newShape, resizeMode mode = resizeMode::eDontAllowResize) {
		auto p = newShape.volume();
		if (p == 0)
			throw std::invalid_argument("Shapes with zero volume are not allowed.");
		
		if (data.isnull()) {
			if (mode == resizeMode::eForceResize) {
				data.resize(p);
			}
		}
		else {
			if (p != data.size()) {
				if (mode == resizeMode::eDontAllowResize)
					throw shapeError("The new shape requires a resize but is not allowed.", data.size(), newShape);
				data.resize(p);
			}
		}

		shape = newShape;
		updateStridesFromShape();
	}
	void setStride(tensorShape const& newShape, tensorShape const& newStrides, int newOffset = 0) {
		shape = newShape;
		strides = newStrides;
		offset = newOffset;
		checkShapeStrideConsistency();
		defaultStride = false; // TODO Check if the given stride is in fact the natural stride.
	}
	inline tensorShape getShape() const  {
		return shape;
	}
	inline tensorShape getStride() const {
		return strides;
	}
	inline int getOffset() const {
		return offset;
	}
	inline int getDataSize() const  {
		return data.size();
	}

	//// Returns the total sum of all elements.
	//float total() const {
	//	float ret = 0;
	//	for (int i = 0; i < getLinearSize(); ++i)
	//		ret += data[i];
	//	return ret;
	//}

	// ===== Basic access =====

	// Tensor access:
	constexpr inline auto& operator()(this auto&& self, index_t const& idx) {
		throw_if_out_of_bounds(self.shape, idx, "");
		return self.data[self.toLinear(idx)];
	}
	template<std::integral... INT>
	constexpr inline auto& operator()(this auto&& self, INT const&...idx) {
		return self(index_t{ idx... });
	}

	// Linear access:
	constexpr inline auto& operator[](this auto&& self, int idx) {
		if (self.defaultStride) {
			return self.data[idx];
		}
		else {
			// Unravel with shape, then ravel with stride:
			int didx = self.offset;
			int d = self.shape.size();
			for (int i = d - 1, ps = 1; i >= 0; --i) {
				didx += self.strides[i] * ((idx / ps) % self.shape[i]); // TODO precompute ps
				ps *= self.shape[i];
			}
			return self.data[didx];
		}
	}

	// Data manipulation:
	TensorData getData() const {  // TODO make TensorData cv qualifier apply to its data
		return data;
	}
	void setData(TensorData&& td) {
		if (!td.isnull() && td.size() != shape.volume())
			throw shapeError("Volumes must match.", shape, tensorShape{ td.size() });
		data = std::forward<TensorData>(td);
	}
	TensorData detachData() {
		TensorData temp = std::move(data);
		data = TensorData();
		return temp;
	}

	void fill(initializer const& init) {
		auto stateful = init;
		int v = shape.volume();
		for (int i = 0; i < v; ++i)
			(*this)[i] = getValue(stateful);
	}

	auto begin(this auto&& self) {
		return TensorIterator<false>{ &self,0 };
	}
	auto end(this auto && self) {
		return TensorIterator<false>{ &self,self.shape.volume() };
	}

	// ===== Advanced access =====
	
	// Returns a view (borrowed data) on this tensor.
	const Tensor view(tensorShape const& shape) const {
#if defined _DEBUG
		if (data.isnull())
			throw std::runtime_error("View on empty tensor not allowed."); // TODO is this a good restriction?
#endif
		Tensor ret = *this;
		ret.reshape(shape);
		return ret;
	}
	const Tensor view(tensorShape const& shape, tensorShape const& stride, int offset = 0) const {
#if defined _DEBUG
		if (data.isnull())
			throw std::runtime_error("View on empty tensor not allowed."); // TODO is this a good restriction?
#endif
		Tensor ret = *this;
		ret.setStride(shape, stride, offset);
		return ret;
	}


	// Copies the values TODO combine with operator=?
	void assign(Tensor const& r) {
		for (int i = 0; i < shape.volume(); ++i)
			(*this)[i] = r[i];
	}

	// Returns a view on the sub-tensor for which the first index is idx0.
	Tensor slice(int idx0) const {
		auto s = shape;
		int m = s.pop_front();
		int n = s.volume();
		if (idx0 < 0) idx0 += m;
		return Tensor(s, data.subset(idx0 * n, n)); // TODO respect stride
	}

	// Returns a view on the sub-tensor for which the first index is idx0.
	const Tensor take(int begin, int end) const {
		auto s = shape;
		s[0] = end-begin;
		int n = s.tail().volume();
		return Tensor(s, data.subset(begin * n, n*(end-begin))); // TODO respect stride
	}

	// Returns a view with all one-dimensions removed.
	const Tensor squeeze() const {
		Tensor ret = *this;
		auto& s = ret.shape, &st = ret.strides;
		
		int remValue = 1;
		auto first = std::find(s.begin(), s.end(), remValue);
		auto f2 = st.begin() + std::distance(s.begin(), first);
		if (first != s.end())
			for (auto i = first; ++i != s.end();)
				if (!(*i == remValue)) {
					*first = std::move(*i); ++first;
					*f2 = st[i-s.begin()]; ++f2;
				}
		s.setNDims(std::distance(s.begin(), first));
		st.setNDims(std::distance(st.begin(), f2));

		return ret;
	}

	// Returns a view with a one-dimension added in dimension idx.
	const Tensor unsqueeze(int idx) const {
		Tensor ret = *this;
		auto& s = ret.shape, &st = ret.strides;
		s.push_back(0);
		st.push_back(0);
		int p = 1;
		for (int i = s.size() - 2; i >= idx; --i){
			s[i + 1] = s[i];
			st[i + 1] = st[i];
			p *= s[i];
		}
		s[idx] = 1;
		st[idx] = p;
		return ret;
	}

	// Returns a view with all dimensions reversed
	const Tensor reverse() const {
		auto s = strides;
		for (int i = 0; i < s.size(); ++i)
			s[i] = -s[i];
		int o = shape.volume() - 1;
		return view(shape, s, o);
	}
	const Tensor reverse(int begin, int end) const {
		auto s = strides;
		int v = 1;
		int o = 0; // TODO, old offset?
		for (int i = s.size()-1; i >= 0; --i) {
			if (i < end && i >= begin) {
				o += (shape[i] - 1) * v;
				s[i] = -s[i];
			}
			v *= shape[i];
		}
		
		return view(shape, s, o);
	}

	// ===== Convenience operators =====
	
	// Returns whether all elements compare equal.
	bool operator==(Tensor const& other) const {
		if (shape != other.shape) return false;
		int n = shape.volume();
		for (int i = 0; i < n; ++i) {
			float a = (*this)[i], b = other[i];
			if (a != b)
				return false;
		}
		return true;
	}

	// ===== Output =====

	// Numpy-like output.
	friend std::ostream& operator<<(std::ostream& os, Tensor const& ten) {
		return os << ten.toString([](float value){return fmt::format("{:> 6.2f}", value); }, ", ");
	}

	// Coloured output.
	void plot(std::pair<float, float> range = { 0,0 }) const {
		int n = shape.volume();
		int d = shape.size();
		tensorShape fa;
		for (int x = 1, j = d - 1; j >= 0; --j)
			fa.push_back(x *= shape[j]);

		auto cf = [](float t) {
			t = std::clamp(t, 0.f, 1.f);
			// Rainbow values
			const float rgb[9][3] = {
				{0.471412, 0.108766, 0.527016},
				{0.248829, 0.264202, 0.782561},
				{0.288863, 0.543185, 0.766607},
				{0.419933, 0.695265, 0.555374},
				{0.621563, 0.743557, 0.350099},
				{0.824105, 0.699564, 0.250430},
				{0.902328, 0.490782, 0.199022},
				{0.857359, 0.131106, 0.132128},
			};

			int i = t * 7;
			float w = t * 7 - i;
			return fmt::rgb(
				std::lerp(rgb[i][0], rgb[i + 1][0], w) * 255,
				std::lerp(rgb[i][1], rgb[i + 1][1], w) * 255,
				std::lerp(rgb[i][2], rgb[i + 1][2], w) * 255);
		};

		auto [minValue, maxValue] = range;
		if (minValue >= maxValue) {
			minValue = std::numeric_limits<float>::max(), maxValue = -minValue;
			for (int i = 0; i < n; ++i) {
				minValue = std::min(minValue, data[i]);
				maxValue = std::max(maxValue, data[i]);
			}
		}
		
		fmt::print("{}\n", toString([&](float value) {
			float t = (value - minValue) / (maxValue - minValue + 1e-6);
			return fmt::format(fg(cf(t)), t < 0 ? "- " : t > 1 ? "+ " : "██");// "\u25A6");
			}, ""));
	}

	// Save to file/stream.
	void serialize(std::ostream& os) const {
		os << shape.size() << ' ';
		for (int i = 0; i < shape.size(); ++i)
			os << shape[i] << ' ';
		assert(data.size() == shape.volume());
		os.write((char*)data.data(), data.size() * sizeof(float));
		os.flush();
	}
	void serialize(fs::path filename) const {
		if(filename.has_parent_path())
			fs::create_directories(filename.parent_path());
		std::ofstream file(filename, std::ios::binary);
		serialize(file);
		file.close();
	}

	// Load from file/stream. Returns if successful.
	bool deserialize(std::istream& is, bool allowResize = true) {
		if (is.peek() == std::ifstream::traits_type::eof())
			return false;

		int d = readNumber(is);
		tensorShape newShape = {};
		for (int i = 0; i < d; ++i) {
			int s = readNumber(is);
			newShape.push_back(s);
		}

		if (!allowResize && shape != newShape)
			throw shapeError("Deserialized wrong shape.", shape, newShape);

		reshape(newShape, resizeMode::eForceResize);
		is.read((char*)data.data(), data.size() * sizeof(float));
		return is.good();
	}
	void deserialize(fs::path filename, bool allowResize = true) {
		if (!fs::exists(filename))
			throw std::runtime_error("File not found: " + filename.string());
		std::ifstream file(filename, std::ios::binary);
		deserialize(file, allowResize);
		file.close();
	}
	

	inline bool getIsDefaultStrided() const {
		return defaultStride;
	}

	
};

template<>
auto& TensorIterator<>::operator*() {
	return (*data)[i];
}


// fmt support
template<> struct fmt::formatter<Tensor> : formatter<std::string> {
	auto format(Tensor const& ten, format_context& ctx) {
		auto s = ten.toString([](float value) {return fmt::format("{:> 6.2f}", value); }, ", ");
		return formatter<std::string>::format(s, ctx);
	}
};


void copy(Tensor src, Tensor dst) {
	throw_if_shape_mismatch(dst.getShape(), src.getShape(), "copy(): The source shape should be destination shape.");
	int n = dst.getShape().volume();
	for (int i = 0; i < n; ++i) {
		dst[i] = src[i];
	}
}

// Creates a new one-dimensional tensor.
Tensor arange(int num, int firstValue = 0) {
	Tensor t(tensorShape{ num });
	auto d = t.getData();
	std::iota(d.data(), d.data() + d.size(), firstValue);
	return t;
}

// Returns a copy of the transpose of t.
Tensor tranpose(Tensor const& t) {
	// TODO consider stride
	auto s = t.getShape(), st = t.getStride();
	std::reverse(s.begin(), s.end());
	std::reverse(st.begin(), st.end());
	auto tt = t.view(s,st);
	Tensor ret(s);
	copy(tt, ret); 
	return ret;
}



// Creates a new tensor and copies the old data into it.
Tensor dilate(Tensor const& t, shapeOrScalar const& dilation) {
	auto s = t.getShape();
	int d = s.size();
	int n = s.volume();
	tensorShape dil = getShape(dilation, d);
	throw_if_ndims_mismatch(d, dil, "dilate(): The dilation should have the same dimensionality as the tensor.");
	for (int i = 0; i < d; ++i)
		s[i] += (s[i] - 1) * dil[i];

	Tensor r(s);
	tensorShape stride = r.getStride();
	for (int i = 0; i < d; ++i) stride[i] *= dil[i] + 1;
	Tensor core = r.view(t.getShape(), stride);

	// Copy the original tensor data to the core part.
	copy(t, core);

	return r;
}


// Creates a new tensor and copies the old data into it.
Tensor dilateR(Tensor const& t,shapeOrScalar const& dilation) {
	auto s = t.getShape();
	int d = s.size();
	int n = s.volume();
	tensorShape dil = getShape(dilation,d);
	throw_if_ndims_mismatch(d,dil,"dilate(): The dilation should have the same dimensionality as the tensor.");
	for (int i = 0; i < d; ++i)
		s[i] += s[i] * dil[i];

	Tensor r(s);
	tensorShape stride = r.getStride();
	for (int i = 0; i < d; ++i) stride[i] *= dil[i] + 1;
	Tensor core = r.view(t.getShape(),stride);

	// Copy the original tensor data to the core part.
	copy(t,core);

	return r;
}

// Creates a new tensor and copies the old data into it. The balance is added to the left padding.
Tensor pad(Tensor const& t, shapeOrScalar const& totPadding, shapeOrScalar const& balance = 0) {
	auto s = t.getShape();
	int d = s.size();
	int n = s.volume();
	tensorShape tp = getShape(totPadding, d);
	tensorShape b = getShape(balance,d);
	throw_if_ndims_mismatch(d, tp, "pad(): The padding should have the same dimensionality as the tensor.");
	for (int i = 0; i < d; ++i)
		s[i] += tp[i];

	Tensor r(s);
	tensorShape stride = r.getStride();
	int offset = 0; for (int i = 0; i < d; ++i) offset += stride[i] * (std::floor(tp[i]/2.f) + b[i]);
	Tensor core = r.view(t.getShape(), stride, offset);

	// Copy the original tensor data to the core part.
	copy(t, core);

	return r;
}



// Returns the linear index of the highest value of t.
int argmax(Tensor const& t) {
	auto d = t.getData();
	return std::max_element(d.data(), d.data() + d.size()) - d.data();// TODO respect stride
}
