// based on https://github.com/mre/hyperjson/blob/master/src/lib.rs

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyFloat, PyList, PyTuple},
};
use serde::{
    ser::{SerializeMap as _, SerializeSeq as _},
    Serialize, Serializer,
};

pub struct SerializePyObject<'a> {
    v: Bound<'a, PyAny>,
}

impl<'a> SerializePyObject<'a> {
    pub fn new(v: Bound<'a, PyAny>) -> Self {
        SerializePyObject { v }
    }
}

pub fn to_json_value(v: Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    let obj = SerializePyObject::new(v);
    serde_json::to_value(&obj).map_err(|e| PyValueError::new_err(format!("{}", e)))
}

#[allow(dead_code)]
pub fn to_json_string(v: Bound<'_, PyAny>) -> PyResult<String> {
    let obj = SerializePyObject::new(v);
    serde_json::to_string(&obj).map_err(|e| PyValueError::new_err(format!("{}", e)))
}

pub fn stringify_if_needed(v: Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = v.extract::<String>() {
        return Ok(s);
    }
    let obj = SerializePyObject::new(v);
    serde_json::to_string(&obj).map_err(|e| PyValueError::new_err(format!("{}", e)))
}

#[allow(dead_code)]
pub fn str_or_dict_to_value(v: Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if let Ok(s) = v.extract::<String>() {
        return serde_json::from_str(&s).map_err(|e| PyValueError::new_err(format!("{}", e)));
    }
    to_json_value(v)
}

impl Serialize for SerializePyObject<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        macro_rules! extract {
            ($t:ty) => {
                if let Ok(val) = self.v.extract::<$t>() {
                    return val.serialize(serializer);
                }
            };
        }

        fn debug_py_err<E: serde::ser::Error>(err: PyErr) -> E {
            E::custom(format_args!("{:?}", err))
        }

        extract!(String);
        extract!(bool);

        if let Ok(x) = self.v.downcast::<PyFloat>() {
            return x.value().serialize(serializer);
        }

        extract!(u64);
        extract!(i64);

        if self.v.is_none() {
            return serializer.serialize_unit();
        }

        if let Ok(x) = self.v.downcast::<PyDict>() {
            let mut map = serializer.serialize_map(Some(x.len()))?;
            for (key, value) in x {
                if let Ok(key) = key.str() {
                    let key = key.to_string();
                    map.serialize_key(&key)?;
                } else {
                    return Err(serde::ser::Error::custom(format_args!(
                        "Dictionary key is not a string: {:?}",
                        key
                    )));
                }
                map.serialize_value(&SerializePyObject { v: value })?;
            }
            return map.end();
        }

        if let Ok(x) = self.v.downcast::<PyList>() {
            let mut seq = serializer.serialize_seq(Some(x.len()))?;
            for element in x {
                seq.serialize_element(&SerializePyObject { v: element })?
            }
            return seq.end();
        }

        if let Ok(x) = self.v.downcast::<PyTuple>() {
            let mut seq = serializer.serialize_seq(Some(x.len()))?;
            for element in x {
                seq.serialize_element(&SerializePyObject { v: element })?
            }
            return seq.end();
        }

        match self.v.repr() {
            Ok(repr) => Err(serde::ser::Error::custom(format_args!(
                "Value is not JSON serializable: {}",
                repr,
            ))),
            Err(_) => Err(serde::ser::Error::custom(format_args!(
                "Type is not JSON serializable: {}",
                self.v.get_type().name().map_err(debug_py_err)?
            ))),
        }
    }
}
