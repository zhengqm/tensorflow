
# How tensorflow represents and optimizes a graph

Computation graph is the core concept in tensorflow. This concept is implemented in `tensorflow/core/graph/graph.h`:

```
class Edge;
class Graph;
class Node;
```


Given the representation of a graph, the interface of an optimizer is in `tensorflow/core/common_runtime/graph_optimizer.h`
and an implementation of that interface is in `tensorflow/core/common_runtime/graph_optimizer.c`:
(tensorflow/core/graph)
```
void GraphOptimizer::Optimize(FunctionLibraryRuntime* runtime, Env* env,
                              Device* device, std::unique_ptr<Graph>* graph) {
  Graph* g = graph->get();
  DumpGraph("Initial", g);

  bool changed = true;
  const int kMaxRounds = 10;
  for (int rounds = 0; rounds < kMaxRounds; ++rounds) {
    changed = false;
    if (RemoveListArrayConverter(g)) {
      DumpGraph("RemoveListArrayConverter", g);
      changed = true;
    }
    if (opts_.do_function_inlining() && RemoveDeadNodes(g)) {
      DumpGraph("RemoveDeadNodes", g);
      changed = true;
    }
    if (opts_.do_function_inlining() && RemoveIdentityNodes(g)) {
      DumpGraph("RemoveIdentityNodes", g);
      changed = true;
    }

    if (opts_.do_constant_folding()) {
      ConstantFoldingOptions cf_opts;
      if (DoConstantFolding(cf_opts, runtime, env, device, g)) {
        RemoveDeadNodes(g);
        DumpGraph("ConstFolding", g);
        changed = true;
      }
    }

    if (opts_.do_function_inlining() && FixupSourceAndSinkEdges(g)) {
      DumpGraph("FixupSourceAndSinkEdges", g);
      changed = true;
    }
    if (opts_.do_common_subexpression_elimination() &&
        OptimizeCSE(g, nullptr)) {
      DumpGraph("OptimizeCSE", g);
      changed = true;
    }
    if (opts_.do_function_inlining() && ExpandInlineFunctions(runtime, g)) {
      DumpGraph("ExpandInlineFunctions", g);
      changed = true;
    }
    if (!changed) break;
  }

  // Note that we use the Graph constructor that copies the input
  // FunctionLibraryDefinition, since the original lib def will go out of scope.
  std::unique_ptr<Graph> copy(new Graph(g->flib_def()));
  CopyGraph(*g, copy.get());
  graph->swap(copy);

  DumpGraph("ReCopy", graph->get());
}
```

where techniques like constant folding and common subexpression elimination are used.