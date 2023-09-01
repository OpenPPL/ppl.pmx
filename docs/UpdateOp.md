When updating an operator, it needs to be done in two cases:

1. If you only need to add parameters to the operator, and the logic does not conflict with the original operator. Parameters can be directly added to the operator and default values ​​can be set for compatibility with older versions.

2. If the operator parameters or logic of the new version are not compatible with the old version, add a new operator definition of `OpTypeName-V`+`version number` at the top of the same file. The old definition will not be modified and can be used with the new ones exist at the same time. The old and new version operators are separated by dividing lines.

Example:

```
# <OpTypeName>-V3

## Attributes/Parameters

## Inputs

## Outputs

## Type Constraints

******

# <OpTypeName>-V2

## Attributes/Parameters

## Inputs

## Outputs

## Type Constraints

******

# <OpTypeName>

## Attributes/Parameters

## Inputs

## Outputs

## Type Constraints
```