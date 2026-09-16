"""Caffe Datum protobuf compatibility without requiring the Caffe package."""
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


_file_descriptor = descriptor_pb2.FileDescriptorProto(
    name='caffe_datum.proto',
    package='caffe',
    syntax='proto2',
)
_message = _file_descriptor.message_type.add(name='Datum')

for name, number, field_type in (
    ('channels', 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
    ('height', 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
    ('width', 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
    ('data', 4, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES),
):
    _message.field.add(
        name=name,
        number=number,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
        type=field_type,
    )

_descriptor = descriptor_pool.Default().Add(_file_descriptor)
Datum = message_factory.GetMessageClass(_descriptor.message_types_by_name['Datum'])