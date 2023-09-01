When adding a new operator, add it in `docs/operators` according to the following format. The format and required content are as follows

Example: Refer to [ColumnParallelLinear](operators/ColumnParallelLinear.md)

```
# <OpTypeName>

Description of this operator

1. mathematic definition(latex) or pseudocode
2. shape infer method
3. tensor parallel behavior

## Attributes/Parameters

### `<attr1>`: <datatype>(<default value>)

Description of this attribute

## Inputs

### `<inputname1>`(<tags...>): <type constraint>

Description of this input

1. meaning
2. shape

<tags...> could be:
1. "constant" if input is weight
2. "optional" if input is optional
3. "inout" if input will be modified

## Outputs

### `<outputname1>`(<tags...>): <type constraint>

Description of this output

1. meaning
2. shape

<tags...> could be:
1. "optional" if input is optional

## Type Constraints

### `<TypeName>`: <type1>, <type2>

Description of this type

## Example

Give some example if necessarily.

```