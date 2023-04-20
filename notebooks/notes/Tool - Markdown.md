

## 1.0 References
1. [basic](https://www.markdownguide.org/basic-syntax/)
2. [extended](https://www.markdownguide.org/extended-syntax/)
3. drawing diagrams
	1. [mermaid](https://mermaid.js.org/syntax/flowchart.html)



## 2.0 Components

### Structure
	- Headers
		- # , ##, ### etc

	- Hyphens: -
	- Bullets: *

	- Lists	
		1. item1
		2. item2


### Code blocks
	- Normal: 
```
	Line 1
	Line 2
```

	- Python: 
``` python
x = [ 2 * i for i in range(10)]
```

- Scala
``` scala
val x = Seq.empty[List]()
```



### Links
```
My favorite search engine is [Duck Duck Go](https://duckduckgo.com).
```
My favorite search engine is [Duck Duck Go](https://duckduckgo.com).




### Highlight Important Text
- add === around the text
	- Ex: This word is very ===important===
- add ** around text to bold
	- Ex: This word is **bolded** .  
- add * around text to italic
	- Ex: This word is *italicized*





### Tables
- To add a table, use three or more hyphens (`---`) to create each column’s header, and use pipes (`|`) to separate each column. For compatibility, you should also add a pipe on either end of the row.
	- :-- is to align text to the left
	- :----:  aligns text to the middle
	- ---: aligns text to right

| Syntax      | Description | Test Text     |
| :---        |    :----:   |          ---: |
| Header      | Title       | Here's this   |
| Paragraph   | Text        | And more      |

- Note: linking to custom header in Obsidian is still not working




### Drawing Diagrams With [Mermaid](https://mermaid.js.org/syntax/flowchart.html)

#### Flow Charts


```mermaid
graph LR;
    A--msg1-->B;
    A-->C;
    B-->D;
    C-->D;
```


#### Mind Maps
- Obsidian needs to upgrade to mermaid v10 [ref](https://forum.obsidian.md/t/mermaid-mindmap-and-timeline-feature-not-available-in-obsidian/47125/10), which has mindmap.
```mermaid
mindmap
  root((mindmap))
    Origins
      Long history
      ::icon(fa fa-book)
      Popularisation
        British popular psychology author Tony Buzan
    Research
      On effectiveness<br/>and features
      On Automatic creation
        Uses
            Creative techniques
            Strategic planning
            Argument mapping
    Tools
      Pen and paper
      Mermaid


```



### Images
```
![Tux, the Linux mascot](/assets/images/tux.png)
```

![Tux, the Linux mascot](/assets/images/tux.png)


### Tables of Contents
```

1. [Example](#example)
2. [Example Miscellaneious](#example2)
3. [Example Text](#example%20text)


###### Example
###### Example2
###### Example Text

```
 TOC
1. [Example](#example)
2. [Example Miscellaneious](#example2)
3. [Example Text](#example%20text)


### Markdown for ML
- Works in Obsidian

- integration
$$
  \int_0^\infty \frac{x^3}{e^x-1}\,dx = \frac{\pi^4}{15}
$$

- subscript
$$
  w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
$$ 
- summation
$$
\sum_{i=1}^{10} t_i
$$


- Use one variable to predict another variable		$$\hat{y}$ = $\theta_0 + \theta_1 x_1$$

- Mean square error
		$$MSE = \frac{1}{n} \Sigma{i=1}^n({y}-\hat{y})^2$$

- Logistic Regression
$$J(\Theta) = -\frac{1}{m} \Sigma{i=1}^my^ilog(\hat{y}^i) + (1-y^i)log(1-\hat{y}^i)$$

- Sigmoid
		$$\Sigma(\Theta^TX)$ = $\frac{1} {1 + e^-\Theta^Tx }$$
		$$\hat{y} = $\Sigma(\Theta^T X)$$	
- Cost function
$$Cost(\hat{y}, y) =  $\frac{1}{2}(\Sigma(\theta^TX)-y)^2$$


- References
  - [md for math reference](https://csrgxtu.github.io/2015/03/20/Writing-Mathematic-Fomulars-in-Markdown/)
  - [ML exmpales](https://krish9a.medium.com/mathematical-notations-for-machine-learning-markdown-5feb99e8d412)



