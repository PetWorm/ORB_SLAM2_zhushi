/**
 * File: FeatureVector.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: feature vector
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_FEATURE_VECTOR__
#define __D_T_FEATURE_VECTOR__

#include "BowVector.h"
#include <map>
#include <vector>
#include <iostream>

namespace DBoW2 {

// QXC：该类的对象应当实例化在一个图像帧对象中，first键为BoW树中的一个节点ID，second键为一个vector容器，这个容器内包含了帧内被分类到该BoW节点的特征在该帧特征队列中的ID
// QXC：而且这个类由于是基于map的，因此会按照key的顺序进行自动排序，默认是由小到大（可由ORBmatcher::SearchByBoW中的该类应用印证）
  /// Vector of nodes with indexes of local features
class FeatureVector: 
  public std::map<NodeId, std::vector<unsigned int> >
{
public:

  /**
   * Constructor
   */
  FeatureVector(void);
  
  /**
   * Destructor
   */
  ~FeatureVector(void);
  
  /**
   * Adds a feature to an existing node, or adds a new node with an initial
   * feature
   * @param id node id to add or to modify
   * @param i_feature index of feature to add to the given node
   */
  void addFeature(NodeId id, unsigned int i_feature);

  /**
   * Sends a string versions of the feature vector through the stream
   * @param out stream
   * @param v feature vector
   */
  friend std::ostream& operator<<(std::ostream &out, const FeatureVector &v);
    
};

} // namespace DBoW2

#endif

